import ee
import pandas as pd
import numpy as np

ee.Initialize(project='ee-zcs')

# ============================================================================
# 数据准备
# ============================================================================
# Hansen 数据
HANSEN = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
treecover2000 = HANSEN.select('treecover2000')
lossyear = HANSEN.select('lossyear')
datamask = HANSEN.select('datamask')

analysis_start_year = 2001
analysis_end_year = 2024
forest_threshold = 30

# 初始森林 (2000, 树冠覆盖 >=30%)
initial_forest_2000 = treecover2000.gte(forest_threshold).And(datamask.eq(1))

# 逐年森林损失 band
def create_hansen_annual_deforestation():
    images = []
    for year in range(analysis_start_year, analysis_end_year + 1):
        hansen_year_code = year - 2000
        annual_defor = initial_forest_2000.And(lossyear.eq(hansen_year_code))
        images.append(annual_defor.rename(f'hansen_defor_{year}'))
    return ee.Image.cat(images)

hansen_annual_deforestation = create_hansen_annual_deforestation()

# ============================================================================
# WDPA 掩膜 (1=保护地, 0=非保护地)
# ============================================================================
wcmc_protected_areas = ee.FeatureCollection('WCMC/WDPA/current/polygons')
pa_mask = (
    ee.Image(0).byte()
      .paint(wcmc_protected_areas, 1)
      .rename('pa')
      .unmask(0)
)

# 面积栈：逐年损失面积 (㎡) + 初始森林面积 (㎡)
annual_defor_area_stack = hansen_annual_deforestation.updateMask(initial_forest_2000) \
                           .multiply(ee.Image.pixelArea())
initial_forest_area = initial_forest_2000.multiply(ee.Image.pixelArea()).rename('initial_forest_2000')
area_stack_all = annual_defor_area_stack.addBands(initial_forest_area)

# 两个版本：保护地 / 非保护地
area_stack_prot = area_stack_all.updateMask(pa_mask)
area_stack_nonprot = area_stack_all.updateMask(pa_mask.Not())

# ============================================================================
# 辅助函数
# ============================================================================
def build_buffers(filtered_codf_polygons, buffer_dist):
    """对 CODF 数据做缓冲"""
    def _buf(f):
        return (f.buffer(buffer_dist)
                 .set('buffer_dist_m', buffer_dist)
                 .set('buffer_km', buffer_dist/1000.0)
                 .set('original_BU_ID', f.get('BU_ID'))
                )
    return filtered_codf_polygons.map(_buf)

def reduce_area_stack_over_buffers(buffers_fc, area_stack, tile_scale=4):
    """对缓冲后的要素，reduce Hansen 栈"""
    stats = area_stack.reduceRegions(
        collection=buffers_fc,
        reducer=ee.Reducer.sum(),
        scale=30,
        tileScale=tile_scale  # <--- tileScale 在这里生效
    )
    return stats

def run_by_buffer_distance(filtered_codf_polygons, buffer_dist, export_prefix, folder):
    """对单个 buffer_dist 同时导出 protected / non-protected"""
    buffers_fc = build_buffers(filtered_codf_polygons, buffer_dist)

    # a) Protected
    prot_stats = reduce_area_stack_over_buffers(buffers_fc, area_stack_prot, tile_scale=4)
    task_p = ee.batch.Export.table.toDrive(
        collection=prot_stats,
        description=f'{export_prefix}_PROTECTED_{buffer_dist//1000}km_2001_2024',
        fileFormat='CSV',
        folder=folder
    )
    task_p.start()

    # b) Non-protected
    nonprot_stats = reduce_area_stack_over_buffers(buffers_fc, area_stack_nonprot, tile_scale=4)
    task_np = ee.batch.Export.table.toDrive(
        collection=nonprot_stats,
        description=f'{export_prefix}_NON_PROTECTED_{buffer_dist//1000}km_2001_2024',
        fileFormat='CSV',
        folder=folder
    )
    task_np.start()

def export_forest_analysis_mask_based(filtered_codf_polygons, buffers):
    """主函数：循环所有 buffer 距离"""
    for d in buffers:
        run_by_buffer_distance(
            filtered_codf_polygons,
            buffer_dist=d,
            export_prefix='forest_loss',
            folder='forestAnalysisSimple'
        )
    print("✓ Export tasks started for buffers:", buffers)

# ============================================================================
# 执行示例
# ============================================================================
# CODF 数据 (你需要替换为自己的资产路径)
CODF_polygons = ee.FeatureCollection("projects/ee-zcs/assets/CODF_polygons")
CODF_points = ee.FeatureCollection("projects/ee-zcs/assets/CODF_points")
CODF_lines = ee.FeatureCollection("projects/ee-zcs/assets/CODF_lines")
CODF_combined = CODF_polygons.merge(CODF_lines)

#combine all CODF datasets into a single FeatureCollection
CODF_combined = CODF_polygons.merge(CODF_lines)
print("Combined CODF FeatureCollection:", CODF_combined.size().getInfo())
#filter good sites based on the availability threshold
availability_threshold = 90  # Set your threshold here
output_file = f'/usr2/postdoc/chishan/project_data/CODF/tmf_suitable_sites_90percent.csv'
good_sites = pd.read_csv(output_file)
filtered_CODF_polygons = CODF_combined.filter(ee.Filter.inList('BU ID', good_sites['BU_ID'].tolist()))

# 示例 buffer 列表 (可以改成你的完整列表 [5000,10000,15000,20000,25000,50000])
buffers = [5000, 10000]

# 调用主函数
export_forest_analysis_mask_based(CODF_combined, buffers)

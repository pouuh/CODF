"""Export per-municipality annual loss (Hansen, GLANCE, MapBiomas, TerraClass)

Run this script from a notebook cell or a Python session that has Earth Engine authenticated.
It will create a FeatureCollection with loss properties per municipality and start an
Earth Engine export to Google Drive as a single CSV file.

Configurable options are at the top of the file.
"""
import ee


def main(
    asset_munis='projects/ee-zcs/assets/BR_Municipios_2021',
    out_drive_folder='EE_exports',
    out_file_prefix='municipality_loss_2009_2022',
    years=None,
    tile_scale=4,
):
    """Compute losses per municipality and export each dataset to separate CSV files.

    Parameters
    - asset_munis: Earth Engine asset id for municipalities (FeatureCollection with CD_MUN property)
    - out_drive_folder: Drive folder name where CSV will be placed
    - out_file_prefix: filename prefix for the CSV (will append dataset name)
    - years: iterable of years to compute (per-year reporting). Defaults to 2009..2022
    - tile_scale: reduceRegions tileScale to help large reductions
    """
    ee.Initialize()

    if years is None:
        years = list(range(2009, 2023))

    # Load municipalities FeatureCollection
    shp = ee.FeatureCollection(asset_munis)
    
    blacklist  =['1100130',
                '1100189',
                '1100205',
                '1100338',
                '1302405',
                '1500602',
                '1501725',
                '1502764',
                '1502939',
                '1505031',
                '1505064',
                '1505502',
                '1506187',
                '1506583',
                '1506708',
                '1507300',
                '1508126',
                '5100250',
                '5101407',
                '5101902',
                '5103254',
                '5103353',
                '5103379',
                '5103858',
                '5105101',
                '5105150',
                '5105580',
                '5106158',
                '5106240',
                '5106299',
                '5106422',
                '5106802',
                '5107065',
                '5107859',
                '5108600',
                '5108907']
    
    shp = shp.filter(ee.Filter.inList('CD_MUN', blacklist))

    # Prepare constant images
    pixel_area_m2 = ee.Image.pixelArea()

    # Hansen source (lossyear band uses 1->2001)
    hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')
    tc = hansen.select('treecover2000')
    forest2000 = tc.gte(30)
    ly = hansen.select('lossyear')

    # GLANCE image collection
    GLANCE_IC = ee.ImageCollection('projects/GLANCE/DATASETS/V001')

    # MapBiomas collection (Collection 10)
    MB_IMG = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2')
    mb_forest_classes = [1, 3, 4, 5, 6, 49]

    def hansen_loss_image(year):
        code = year - 2000
        loss = ly.eq(code).And(forest2000)
        return loss

    def glance_loss_image(year):
        # GLANCE data only available up to 2019
        if year > 2019:
            return None
        # loss from y to y+1: forest(y-1) -> non-forest(y)
        f_y_col = GLANCE_IC.filterDate(f'{year-1}-01-01', f'{year-1}-12-31').select('LC')
        f_y1_col = GLANCE_IC.filterDate(f'{year}-01-01', f'{year}-12-31').select('LC')
        # Check if collections are empty
        if f_y_col.size().getInfo() == 0 or f_y1_col.size().getInfo() == 0:
            return None
        f_y = f_y_col.mosaic().eq(5)
        f_y1 = f_y1_col.mosaic().eq(5)
        return f_y.And(f_y1.Not())

    def mb_loss_image(year):
        band_prev = f'classification_{year-1}'
        band_curr = f'classification_{year}'
        img_prev = MB_IMG.select(band_prev).remap(mb_forest_classes, [1]*len(mb_forest_classes), 0)
        img_curr = MB_IMG.select(band_curr).remap(mb_forest_classes, [1]*len(mb_forest_classes), 0)
        return img_prev.And(img_curr.Not())

    def terra_loss_image_even(year_even):
        # TerraClass assets are stored as projects/ee-zcs/assets/AMZ{year}M and represent land cover
        asset_id = f'projects/ee-zcs/assets/AMZ{year_even}M'
        try:
            img = ee.Image(asset_id)
        except Exception:
            # asset probably doesn't exist
            return None
        # conservative forest classes for TerraClass (adjust as needed)
        terra_forest = img.remap([1, 2], [1, 1], 0)
        return terra_forest

    tasks = []

    # ========== Hansen Export ==========
    print('Processing Hansen dataset...')
    fc_hansen = shp
    for y in years:
        print(f'  Hansen year {y}')
        loss_hans = hansen_loss_image(y).unmask(0)
        area_hans = loss_hans.multiply(pixel_area_m2)
        res_hans = area_hans.reduceRegions(collection=shp, reducer=ee.Reducer.sum(), scale=30, tileScale=tile_scale)
        # Add year column and loss_km2 column
        res_hans = res_hans.map(lambda f: f.set('year', y).set('loss_km2', ee.Number(f.get('sum')).divide(1e6)))
        # Join to main collection
        prop = f'Hansen_{y}'
        dict_hans = ee.Dictionary.fromLists(
            ee.List(res_hans.aggregate_array('CD_MUN')).map(lambda k: ee.String(k)),
            res_hans.aggregate_array('loss_km2')
        )
        fc_hansen = fc_hansen.map(lambda f: f.set(prop, dict_hans.get(ee.String(f.get('CD_MUN')))))

    task_hansen = ee.batch.Export.table.toDrive(
        collection=fc_hansen,
        description=f'{out_file_prefix}_Hansen',
        folder=out_drive_folder,
        fileNamePrefix=f'{out_file_prefix}_Hansen',
        fileFormat='CSV',
    )
    task_hansen.start()
    tasks.append(task_hansen)
    print(f'Hansen export started. Task id: {task_hansen.id}')

    # ========== GLANCE Export ==========
    print('Processing GLANCE dataset...')
    fc_glance = shp
    for y in years:
        print(f'  GLANCE year {y}')
        try:
            loss_gl_img = glance_loss_image(y)
            if loss_gl_img is None:
                print(f'  GLANCE data not available for {y}, skipping')
                continue
            loss_gl = loss_gl_img.unmask(0)
            area_gl = loss_gl.multiply(pixel_area_m2)
            res_gl = area_gl.reduceRegions(collection=shp, reducer=ee.Reducer.sum(), scale=30, tileScale=tile_scale)
            prop = f'GLANCE_{y}'
            dict_gl = ee.Dictionary.fromLists(
                ee.List(res_gl.aggregate_array('CD_MUN')).map(lambda k: ee.String(k)),
                ee.List(res_gl.aggregate_array('sum')).map(lambda v: ee.Number(v).divide(1e6))
            )
            fc_glance = fc_glance.map(lambda f: f.set(prop, dict_gl.get(ee.String(f.get('CD_MUN')))))
        except Exception as e:
            print(f'  GLANCE failed for {y}: {e}')

    task_glance = ee.batch.Export.table.toDrive(
        collection=fc_glance,
        description=f'{out_file_prefix}_GLANCE',
        folder=out_drive_folder,
        fileNamePrefix=f'{out_file_prefix}_GLANCE',
        fileFormat='CSV',
    )
    task_glance.start()
    tasks.append(task_glance)
    print(f'GLANCE export started. Task id: {task_glance.id}')

    # ========== MapBiomas Export ==========
    print('Processing MapBiomas dataset...')
    fc_mb = shp
    for y in years:
        print(f'  MapBiomas year {y}')
        try:
            loss_mb = mb_loss_image(y).unmask(0)
            area_mb = loss_mb.multiply(pixel_area_m2)
            res_mb = area_mb.reduceRegions(collection=shp, reducer=ee.Reducer.sum(), scale=30, tileScale=tile_scale)
            prop = f'MapBiomas_{y}'
            dict_mb = ee.Dictionary.fromLists(
                ee.List(res_mb.aggregate_array('CD_MUN')).map(lambda k: ee.String(k)),
                ee.List(res_mb.aggregate_array('sum')).map(lambda v: ee.Number(v).divide(1e6))
            )
            fc_mb = fc_mb.map(lambda f: f.set(prop, dict_mb.get(ee.String(f.get('CD_MUN')))))
        except Exception as e:
            print(f'  MapBiomas failed for {y}: {e}')

    task_mb = ee.batch.Export.table.toDrive(
        collection=fc_mb,
        description=f'{out_file_prefix}_MapBiomas',
        folder=out_drive_folder,
        fileNamePrefix=f'{out_file_prefix}_MapBiomas',
        fileFormat='CSV',
    )
    task_mb.start()
    tasks.append(task_mb)
    print(f'MapBiomas export started. Task id: {task_mb.id}')

    # ========== TerraClass Export ==========
    print('Processing TerraClass dataset...')
    fc_terra = shp
    for y in years:
        if y % 2 == 0:
            print(f'  TerraClass year {y}')
            try:
                img_curr = terra_loss_image_even(y)
                img_prev = terra_loss_image_even(y-2)
                if img_curr is None or img_prev is None:
                    print(f'  TerraClass assets missing for {y}')
                else:
                    loss_te = img_prev.And(img_curr.Not()).unmask(0)
                    area_te = loss_te.multiply(pixel_area_m2)
                    res_te = area_te.reduceRegions(collection=shp, reducer=ee.Reducer.sum(), scale=30, tileScale=tile_scale)
                    prop = f'TerraClass_{y}'
                    dict_te = ee.Dictionary.fromLists(
                        ee.List(res_te.aggregate_array('CD_MUN')).map(lambda k: ee.String(k)),
                        ee.List(res_te.aggregate_array('sum')).map(lambda v: ee.Number(v).divide(1e6))
                    )
                    fc_terra = fc_terra.map(lambda f: f.set(prop, dict_te.get(ee.String(f.get('CD_MUN')))))
            except Exception as e:
                print(f'  TerraClass failed for {y}: {e}')

    task_terra = ee.batch.Export.table.toDrive(
        collection=fc_terra,
        description=f'{out_file_prefix}_TerraClass',
        folder=out_drive_folder,
        fileNamePrefix=f'{out_file_prefix}_TerraClass',
        fileFormat='CSV',
    )
    task_terra.start()
    tasks.append(task_terra)
    print(f'TerraClass export started. Task id: {task_terra.id}')

    print('\n' + '='*60)
    print('All exports started successfully!')
    print(f'Total tasks: {len(tasks)}')
    print('Check the Earth Engine Tasks tab or use ee.batch.Task.list() to monitor progress.')
    print('='*60)
    
    return tasks


if __name__ == '__main__':
    # When run as a script, start the export with defaults
    main()

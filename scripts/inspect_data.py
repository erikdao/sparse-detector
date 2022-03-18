"""
Ad-hoc script to inspect the results
"""
from cgitb import reset
import os
import sys
import json
from pathlib import Path

import click
import pandas as pd

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)


@click.command()
@click.option("--category-name", default="sheep")
@click.option("--objects-count", default=None)
@click.option("--image-id", type=int, default=None)
def main(category_name, objects_count, image_id):
    anno_dir = Path("./data/COCO/annotations")
    assert anno_dir.exists()

    anno_file = anno_dir / "instances_val2017.json"
    click.echo("Reading val labels...")
    with open(anno_file, "r") as f:
        data = json.load(f)
    
    click.echo("Getting data for the sheep category...")
    cat = list(filter(lambda d: d["name"] == category_name, data["categories"]))[0]
    cat_id = cat['id']
    click.echo(cat)

    click.echo(f"Getting annotations of all images which {category_name} is present...")
    annos = list(filter(lambda d: d["category_id"] == cat_id, data["annotations"]))
    click.echo(f"There are {len(annos)} annotations")
    image_ids = set([anno['image_id'] for anno in annos])
    click.echo(f"... of {len(image_ids)} images")
    
    anno_meta = [dict(
        id=anno['id'], image_id=anno['image_id'], category_id=anno['category_id']
    ) for anno in annos]

    df = pd.DataFrame(anno_meta)
    stats = df.groupby("image_id")\
        .agg(objects_count=('id', 'count'))\
        .sort_values(by=['objects_count'], ascending=False)\
        .reset_index()

    click.echo(stats.head())

    if objects_count is not None:
        click.echo(f"List of images where there are {objects_count} instances of {category_name}")
        filtered_df = stats.query(f'objects_count == {objects_count}')
        click.echo(filtered_df.head(10))

    if image_id is not None:
        images = data['images']
        image = list(filter(lambda d: d['id'] == image_id, images))[0]
        print(image)


if __name__ == "__main__":
    main()

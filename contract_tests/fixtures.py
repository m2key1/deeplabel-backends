import io
import json

from PIL import Image, ImageDraw

INK = (40, 30, 20)
PAPER = (245, 240, 228)


def png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format='PNG')
    return out.getvalue()


def page(seed: int = 0, size: tuple[int, int] = (1200, 1600)) -> Image.Image:
    """A page with a heading and a paragraph of ruled ink bars standing in for text."""
    image = Image.new('RGB', size, PAPER)
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.rectangle([width * 0.3, height * 0.07, width * 0.7, height * 0.11], fill=INK)
    for row in range(14):
        top = height * (0.22 + row * 0.045)
        right = width * (0.88 - 0.03 * ((row + seed) % 4))
        draw.rectangle([width * 0.12, top, right, top + height * 0.018], fill=INK)
    return image


def line_crop(text: str) -> Image.Image:
    image = Image.new('RGB', (40 + 18 * len(text), 64), PAPER)
    ImageDraw.Draw(image).text((20, 18), text, fill=INK, font_size=28)
    return image


def map_tile(size: int = 512) -> Image.Image:
    image = Image.new('RGB', (size, size), PAPER)
    draw = ImageDraw.Draw(image)
    draw.polygon([(60, 60), (260, 80), (240, 260), (70, 240)], outline=INK, width=4)
    draw.line([(0, 400), (size, 360)], fill=INK, width=10)
    return image


def task_images(task: str) -> list[Image.Image]:
    if task == 'ocr':
        return [line_crop('Anno 1754 den 3. May'), line_crop('Herr Johann Weber')]
    if task in ('map_segmentation', 'edges'):
        return [map_tile(), map_tile(384)]
    return [page(0), page(1, (1000, 1400))]


def task_params(task: str, detail: dict) -> dict:
    if task == 'edges':
        method = (detail.get('methods') or [{'id': 'centreline'}])[0]['id']
        return {'method': method, 'area': [[0.1, 0.1], [0.6, 0.1], [0.6, 0.5], [0.1, 0.5]]}
    if task == 'map_segmentation' and detail.get('classes'):
        return {'class_ids': [c['id'] for c in detail['classes']]}
    return {}


def task_context(task: str, index: int) -> dict:
    return {'element_class': 'Paragraph', 'line_index': index} if task == 'ocr' else {}


def _coco(names: list[str], boxes: dict[str, list[list[int]]]) -> bytes:
    categories = [{'id': i, 'name': name} for i, name in enumerate(names)]
    images, annotations = [], []
    for image_id, (file_name, image_boxes) in enumerate(boxes.items()):
        images.append({'id': image_id, 'file_name': file_name, 'width': 1200, 'height': 1600})
        for box in image_boxes:
            annotations.append({'id': len(annotations), 'image_id': image_id, 'category_id': 0, 'bbox': box,
                                'area': box[2] * box[3], 'iscrowd': 0})
    return json.dumps({'images': images, 'annotations': annotations, 'categories': categories}).encode()


def snapshot(kind: str) -> tuple[dict[str, bytes], dict]:
    """A tiny training snapshot of one canonical kind (section 6.3): its files and the
    manifest fields besides the file list."""
    files: dict[str, bytes] = {}
    groups: dict[str, str] = {}
    task = {'ocr-lines/1': 'ocr', 'page-lines/1': 'lines', 'page-boxes/1': 'layout',
            'map-masks/1': 'map_segmentation'}[kind]
    splits = {'train': 6, 'valid': 2}
    for split, count in splits.items():
        for n in range(count):
            stem = f'{split}{n}'
            group = 'dataset-a' if n % 2 == 0 else 'dataset-b'
            if kind == 'ocr-lines/1':
                text = f'Zeile {n} im Jahr 17{50 + n}'
                files[f'{split}/line_{stem}.png'] = png(line_crop(text))
                files[f'{split}/line_{stem}.txt'] = text.encode()
                groups[f'line_{stem}'] = group
            elif kind == 'page-lines/1':
                files[f'{split}/images/{stem}.png'] = png(page(n))
                lines = [{'id': row, 'element_class': 'Paragraph',
                          'polygon': [[144, 352 + 72 * row], [1056, 352 + 72 * row],
                                      [1056, 381 + 72 * row], [144, 381 + 72 * row]]} for row in range(14)]
                files[f'{split}/lines/{stem}.json'] = json.dumps({'width': 1200, 'height': 1600, 'lines': lines}).encode()
                groups[stem] = group
            elif kind == 'page-boxes/1':
                files[f'{split}/images/{stem}.png'] = png(page(n))
                groups[stem] = group
            else:
                tile = map_tile()
                files[f'{split}/images/{stem}.png'] = png(tile)
                mask = Image.new('L', tile.size, 0)
                ImageDraw.Draw(mask).polygon([(60, 60), (260, 80), (240, 260), (70, 240)], fill=1)
                files[f'{split}/annotations/{stem}.png'] = png(mask)
                files[f'{split}/annotations/{stem}.json'] = json.dumps({'schema': 2, 'regions': []}).encode()
                groups[stem] = group
        if kind == 'page-boxes/1':
            boxes = {f'{split}{n}.png': [[360, 112, 480, 64], [144, 352, 912, 1008]] for n in range(count)}
            files[f'{split}/_annotations.coco.json'] = _coco(['Header', 'Paragraph'], boxes)
    if kind == 'map-masks/1':
        files['class_map.json'] = json.dumps({'0': 'background', '1': 'Parcel'}).encode()
    files['counts.json'] = json.dumps(splits).encode()
    manifest = {'format': 'deeplabel-snapshot/1', 'kind': kind.split('/')[0], 'kind_version': 1, 'task': task,
                'training_id': 0, 'splits': splits, 'groups': groups,
                'classes': ['Header', 'Paragraph'] if kind == 'page-boxes/1' else []}
    return files, manifest

from datasets import load_dataset
from PIL import Image

ds = load_dataset("howard-hou/COCO-Text", split="validation")
#filter out rows where len(ds[row]['ocr_info'])==1
#ds = ds.filter(lambda x: len(x['ocr_info']) > 1)
print(ds)
#for each row, save a mask of the image with the bounding_box masked. reach ds[row] looks like {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x427 at 0x7F32688ACE30>, 'coco_file_name': 'COCO_train2014_000000061133.jpg', 'image_id': '61133', 'caption': ['A bottle of milk next to a tray of cookies and red pears.', 'A bottle of fresh milk sits beside a container of chocolate cookies and 3 tangerines on a table. ', 'You have a choice of either milk and cookies or nutritious fruit for a snack.', 'A bottle of milk near some cookies that are chocolaty.', 'There is a glass bottle of milk and some cookies'], 'ocr_tokens': ['CREAMERY', 'BROOK', 'TWIN', '20'], 'ocr_info': [{'word': 'CREAMERY', 'bounding_box': {'width': 0.077, 'height': 0.0419, 'top_left_x': 0.643, 'top_left_y': 0.566}}, {'word': 'BROOK', 'bounding_box': {'width': 0.0481, 'height': 0.0553, 'top_left_x': 0.6809, 'top_left_y': 0.4536}}, {'word': 'TWIN', 'bounding_box': {'width': 0.038, 'height': 0.0419, 'top_left_x': 0.64, 'top_left_y': 0.455}}, {'word': '20', 'bounding_box': {'width': 0.025, 'height': 0.0253, 'top_left_x': 0.903, 'top_left_y': 0.9632}}], 'image_width': 640, 'image_height': 427} :
# 
counter = 0
captions = []
for row in ds:
    image_width = row['image_width']
    image_height = row['image_height']
    image = row['image']
    image.save(f"{counter}_src.png")  # Save the image
    ocr_info = row['ocr_info']
    if(len(ocr_info) != 1):
        continue
    for i, info in enumerate(ocr_info):
        
        bounding_box = info['bounding_box']
        width = int(bounding_box['width'] * image_width)
        height = int(bounding_box['height'] * image_height)
        top_left_x = int(bounding_box['top_left_x'] * image_width)
        top_left_y = int(bounding_box['top_left_y'] * image_height)
        bottom_right_x = top_left_x + width
        bottom_right_y = top_left_y + height
        mask = Image.new('L', (image_width, image_height), 0)
        mask.paste(255, (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
        mask.save(f"{counter}_mask.png") 
        captions.append(info['word'])
        counter += 1
    break

# Save the captions to a file
with open("captions.txt", "w") as f:
    for caption in captions:
        f.write(caption + "\n")
        
   

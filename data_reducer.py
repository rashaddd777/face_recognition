import os, random, shutil

TRAIN_DIR = "data/public/train"
NUM_PEOPLE = 100
IMAGES_PER_PERSON = 100

people = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
keep_people = set(random.sample(people, min(NUM_PEOPLE, len(people))))

for p in people:
    if p not in keep_people:
        shutil.rmtree(os.path.join(TRAIN_DIR, p))

for person in keep_people:
    folder = os.path.join(TRAIN_DIR, person)
    images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if len(images) > IMAGES_PER_PERSON:
        to_keep = set(random.sample(images, IMAGES_PER_PERSON))
        for img in images:
            if img not in to_keep:
                os.remove(os.path.join(folder, img))

print("Pruning complete: 100 people × 100 images each.")

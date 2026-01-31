import shutil
import os

model_dir = r"E:\local first personal AI assistant\backend\models\intent_models\intent_model"
checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
checkpoints.sort(key=lambda x: int(x.split("-")[1]))

best_checkpoint = checkpoints[-1]  # Last one
source = os.path.join(model_dir, best_checkpoint)
dest = os.path.join(model_dir, "final")

print(f"Copying {best_checkpoint} to final/")

if os.path.exists(dest):
    shutil.rmtree(dest)
shutil.copytree(source, dest)

print(f"✅ Model copied to {dest}")
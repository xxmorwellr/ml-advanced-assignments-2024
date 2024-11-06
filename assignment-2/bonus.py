import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModel
import loralib as lora # pip install transformers diffusers accelerate loralib
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as T
from wikiart import WikiArtDataset
from torchvision.transforms import ToTensor
from itertools import islice


# Load pre-trained model
print("Loading pre-trained model...")
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe.to(device)
print("Model loaded and moved to CUDA.")

# Apply LoRA
print("Applying LoRA to attention layers...")
# First, collect attention layers that need modification
layers_to_replace = {}

for name, module in pipe.unet.named_modules():
    if isinstance(module, nn.Linear) and 'attn' in name:
        layers_to_replace[name] = module
    elif isinstance(module, nn.Embedding) and 'attn' in name:
        layers_to_replace[name] = module

# Now, modify the layers after iteration
for name, module in layers_to_replace.items():
    if isinstance(module, nn.Linear):
        pipe.unet._modules[name] = lora.Linear(module.in_features, module.out_features, r=8)
    elif isinstance(module, nn.Embedding):
        pipe.unet._modules[name] = lora.Embedding(module.num_embeddings, module.embedding_dim, r=8)
print("LoRA applied.")

# Mark only LoRA parameters as trainable
lora.mark_only_lora_as_trainable(pipe.unet)

# Load data
imgdir = 'train'
dataset = WikiArtDataset(imgdir, device=device)

# Data preprocessing
def preprocess(image):
    # Resize the image to fit the pre-trained model
    resize_transform = T.Resize((512, 512))
    image = resize_transform(image) 
    
    # # Normalization
    # image = image.float() / 255.0  # [0, 1]
    # image = (image - 0.5) * 2  # [-1, 1]
    
    return image

def collate_fn(batch):
    # Preprocessing in batches
    pixel_values = []
    labels = []

    for imgobj in batch:
        image,_,label = imgobj
        preprocessed_image = preprocess(image)
        pixel_values.append(preprocessed_image)
        labels.append(label)  # string
    
    # Convert to tensors
    pixel_values = torch.stack(pixel_values)
    
    return {"pixel_values": pixel_values, "labels": labels}

train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn) # here `batch_size`=1 

optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=1e-4)
# Fine-tuning
print("Starting fine-tuning...")
for epoch in range(2):
    for batch in islice(train_dataloader, 10): # sliced
        images = batch["pixel_values"].to(device).requires_grad_(True)
        labels = batch["labels"]
        
        # Generate image
        prompt = labels
        output = pipe(prompt).images[0] # PIL Image

        # Convert to tensor
        output = ToTensor()(output).unsqueeze(0).to(device).requires_grad_(True)
        
        loss = torch.nn.functional.mse_loss(output, images) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Generate one test image
with torch.no_grad():
    prompt = "Rococo"  # specific artstyle
    generated_image = pipe(prompt).images[0]

# Save the generated image
generated_image.save("prompted_style_image.png")

# Save the model with only LoRA parameters
# torch.save(lora.lora_state_dict(pipe.unet), 'loRA_finetuned_model.pt')
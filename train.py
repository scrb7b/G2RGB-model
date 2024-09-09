import torch
from torch.distributed import gather_object

from model import model, optimizer, loss_fn
from build_dataset import dataloader
from torchvision import transforms

EPOCHS = 12

total_batches = len(dataloader)

grayscale_transform = transforms.Grayscale()
model.load_state_dict(torch.load('concatcolor3epochs.pth'))
for epoch in range(EPOCHS):
    for i, img in enumerate(dataloader):

        grayscale_image = grayscale_transform(img)


        output = model(grayscale_image)

        #print(img.size(), grayscale_image.size(), output.size())

        loss = loss_fn(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if epoch % 10 == 0:
        progress = (i + 1) / total_batches * 100
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{total_batches}], "
              f"Progress: {progress:.2f}%, Loss: {loss.item()}")

torch.save(model.state_dict(), 'concatcolor15epochs.pth')

print("Training Complete.")
import torch
import wandb
import os
from PIL import Image
from UNetResBlock.model import generate_samples, calc_loss
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(u_net, dataset, config, model_name, log=False, save_model=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u_net = u_net.to(device)

    image_shape = config["image_shape"]
    save_dir = os.path.join("UNetResBlock/results", model_name)  # Create the full directory path
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Optimizer
    opt = torch.optim.Adam(u_net.parameters(), lr=config["learning_rate"])

    # DataLoader
    dloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    results_txt = '/zhome/1a/a/156609/public_html/ShowResults/resultOwn.txt'

    clear_txt_file(results_txt)

    p_bar = True

    for i_epoch in range(config["epochs"]):
        print(f"Epoch {i_epoch} started")
        total_loss = 0

        # Wrap DataLoader with or without tqdm for progress bar
        if p_bar:  # Default to True if not specified
            num_batches = len(dloader)
            progress_bar = tqdm(dloader, total=num_batches, desc=f"Epoch {i_epoch}", ncols=100)
        else:
            progress_bar = dloader  # Use the dataloader directly without tqdm

        # Training loop
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            opt.zero_grad()
            loss = calc_loss(u_net, data)
            loss.backward()
            opt.step()
            total_loss += loss.item() * data.shape[0]

            # Update the description of the progress bar with current loss
            if p_bar:  # Only update tqdm if it's enabled
                progress_bar.set_postfix(loss=loss.item(), total_loss=total_loss)

        avg_loss = total_loss / len(dataset)
        if log:
            wandb.log({"loss": avg_loss}, step = i_epoch)

        torch.save(u_net.state_dict(), f"UNetResBlock/models/{model_name}ckpt.pt")

        if (i_epoch < 5 or i_epoch % 5 == 0):
            print("generating samples")
            # Generate 8 samples after each epoch
            generated_samples = generate_samples(u_net, nsamples=16, image_shape=image_shape, timesteps=1000)
            
            #print_samples_and_data(data, generated_samples)

            # Save the generated samples as a row in the specified folder
            save_samples(generated_samples, save_dir, f"epoch{i_epoch}", i_epoch, wandb_log=True)
            save_samples(generated_samples, "/zhome/1a/a/156609/public_html/ShowResults", "resultOwn")
            
            # Save the first 8 images from the dataset
            first_batch, _ = next(iter(torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)))
            save_samples(first_batch.to(device), save_dir, f"dataset_samples")
            
            add_line_to_txt(results_txt, str(avg_loss))

        # Compute average loss for the epoch
        print(f"Epoch {i_epoch}, Loss: {avg_loss}")

    print("Training complete.")




def save_samples(samples, save_dir, filename, i_epoch=0, wandb_log=False):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Move the samples to CPU if on GPU and scale from [-1, 1] to [0, 1]
    samples = samples.cpu()
    # Compute statistics
    min_val = samples.min().item()
    max_val = samples.max().item()
    avg_val = samples.mean().item()
    std_val = samples.std().item()

    # Print statistics
    print(f"file name: {filename}")
    print(f"with samples min: {min_val:.2f}, samples max: {max_val:.2f}, avg: {avg_val:.2f}, std: {std_val:.2f}")

    samples = (samples + 1) / 2  # Scale from [-1, 1] to [0, 1]
    samples = samples.clamp(0, 1)  # Ensure values are within [0, 1]

    # Save concatenated image as a single PNG file
    images = []
    for i in range(samples.shape[0]):
        img = samples[i].permute(1, 2, 0).clamp(0, 1).numpy() * 255
        if samples.shape[1] == 1:  # Grayscale
            img = img[:, :, 0]
            img = Image.fromarray(img.astype('uint8'), mode='L')
        elif samples.shape[1] == 3:  # RGB
            img = Image.fromarray(img.astype('uint8'))
        else:
            raise ValueError(f"Unexpected number of channels: {samples.shape[1]}")
        images.append(img)

    concatenated_image = Image.new(
        'RGB' if samples.shape[1] == 3 else 'L',
        (samples.shape[0] * images[0].width, images[0].height)
    )

    for i, img in enumerate(images):
        concatenated_image.paste(img, (i * img.width, 0))

    # Save the concatenated image
    save_path = os.path.join(save_dir, f"{filename}.png")
    concatenated_image.save(save_path)

    # Optionally log to wandb with matplotlib for grayscale
    if wandb_log:
        grid_size = 4  # 4x4 grid
        num_images = samples.shape[0]

        if samples.shape[1] == 1:  # Grayscale images
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
            axes = axes.flatten()  # Flatten the 2D grid of axes for easy iteration

            for i, ax in enumerate(axes):
                if i < num_images:
                    img = samples[i][0].cpu().numpy()  # Take the grayscale channel
                    ax.imshow(1 - img, cmap="Greys")
                ax.axis("off")  # Turn off axes for all plots, including blanks

            wandb.log({f"GeneratedSamples:": wandb.Image(fig)}, step = i_epoch)
            plt.close(fig)
        else:  # RGB images
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
            axes = axes.flatten()  # Flatten the 2D grid of axes for easy iteration

            for i, ax in enumerate(axes):
                if i < num_images:
                    img = samples[i].permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
                    ax.imshow((img * 255).astype("uint8"))  # Ensure RGB is properly scaled
                ax.axis("off")  # Turn off axes for all plots, including blanks

            wandb.log({f"GeneratedSamples:": wandb.Image(fig)}, step = i_epoch)
            plt.close(fig)

    #print(f"Saved samples to {save_path}")




def print_samples_and_data(data, generated_samples):
    # Calculate and print dataset statistics
    data_min = data.min().item()
    data_max = data.max().item()
    data_mean = data.mean().item()
    data_std = data.std().item()

    print(f"Dataset Statistics:")
    print(f"  Min: {data_min}, Max: {data_max}")
    print(f"  Avg: {data_mean:.4f}, Std: {data_std:.4f}")

    # Calculate and print generated sample statistics
    samples_min = generated_samples.min().item()
    samples_max = generated_samples.max().item()
    samples_mean = generated_samples.float().mean().item()
    samples_std = generated_samples.float().std().item()

    print(f"Generated Samples Statistics:")
    print(f"  Min: {samples_min}, Max: {samples_max}")
    print(f"  Avg: {samples_mean:.4f}, Std: {samples_std:.4f}")


def add_line_to_txt(file_path, line):
    # Check if the file exists
    if not os.path.exists(file_path):
        # If the file does not exist, create it
        with open(file_path, 'w') as f:
            f.write(line + '\n')  # Write the line and add a newline character
    else:
        # If the file exists, append the line
        with open(file_path, 'a') as f:
            f.write(line + '\n')  # Append the line and add a newline character

def clear_txt_file(file_path):
    with open(file_path, 'w') as f:
        # Opening the file in 'w' mode will clear its contents
        pass  # No need to write anything, just clearing the file
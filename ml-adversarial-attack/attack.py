import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from model import ANNClassifier

class FGSM:
    def __init__(self, model, criterion, epsilon=0.3):
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon

    def fgsm_attack(self, images, labels):
        images.requires_grad = True
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        images_grad = images.grad.data

        images_adv = images + self.epsilon * torch.sign(images_grad)
        images_adv = torch.clamp(images_adv, 0, 1)

        return images_adv

    def apply(self, images, labels):
        self.model.eval()
        total = 0
        correct = 0
        adv_examples = []
        noise_norms = [] 

        device = next(self.model.parameters()).device

        for image, label in zip(images, labels):
            image, label = image.to(device), label.to(device)

            image_adv = self.fgsm_attack(image.unsqueeze(0), label.unsqueeze(0))

            output = self.model(image_adv)
            _, predicted = torch.max(output, 1)
            total += 1
            correct += (predicted == label).item()

            # Reshape the adversarial example to 1*1*28*28
            image_adv = image_adv.view(1, 1, 28, 28)  # Reshape to 1*1*28*28
            adv_examples.append(image_adv.detach().cpu())

            # Calculate noise and its norm
            noise = (image_adv.squeeze() - image.squeeze()).detach().cpu().numpy()
            noise_norm = np.linalg.norm(noise)
            noise_norms.append(noise_norm)

        evasion_rate = 1 - correct / total

        return {"evasion_rate": evasion_rate, "adv_examples": adv_examples, "noise_norms": noise_norms}

def generate_random_samples(X_test, y_test, num_samples_per_digit):
    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    selected_indices = [np.random.choice(indices, num_samples_per_digit, replace=False) for indices in digit_indices]
    selected_indices = np.concatenate(selected_indices)
    X_selected = torch.tensor(X_test[selected_indices], dtype=torch.float32)
    y_selected = torch.tensor(y_test[selected_indices], dtype=torch.long)
    return X_selected, y_selected

def plot_adversarial_examples(original_images, adversarial_images):
    num_digits = len(original_images)
    fig, axes = plt.subplots(3, num_digits // 3, figsize=(num_digits // 3 * 5, 7))

    digit_val = 0

    for i in range(num_digits):
        if (i % 3 != 0):
            continue
        col_index = i // 3  # Calculate the column index for the subplot
        original_img = original_images[i].numpy().reshape(28, 28)  
        adversarial_img = adversarial_images[i].numpy().reshape(28, 28)  
        noise = adversarial_img - original_img  # Calculate the difference between original and adversarial images
        noise_norm = np.linalg.norm(noise)  # Calculate L2 norm of the difference

        axes[0, col_index].imshow(original_img, cmap='gray', aspect='auto')
        axes[1, col_index].imshow(adversarial_img, cmap='gray', aspect='auto')
        axes[2, col_index].imshow(noise, cmap='hot', aspect='auto')  # Plot the difference as heatmap

        axes[0, col_index].axis('off')
        axes[1, col_index].axis('off')
        axes[2, col_index].axis('off')

        axes[0, col_index].set_title(f'Original Digit {digit_val}')
        axes[1, col_index].set_title(f'Adversarial Digit {digit_val}')
        axes[2, col_index].set_title(f'Noise: {noise_norm:.2f}') 
        
        digit_val += 1

    plt.tight_layout()
    plt.show()

def find_most_misclassified_digit(model, fgsm, digit, X_test, y_test):
    # Filter images of the selected digit
    digit_indices = np.where(y_test == digit)[0]
    X_digit = X_test[digit_indices]
    y_digit = y_test[digit_indices]
    
    # Reshape input images to (batch_size, 1, 28, 28)
    X_digit = torch.tensor(X_digit, dtype=torch.float32).view(-1, 1, 28, 28)
    y_digit = torch.tensor(y_digit, dtype=torch.long)
    
    # Apply FGSM attack and classify adversarial examples
    results = fgsm.apply(X_digit, y_digit)
    adversarial_images = results["adv_examples"]
    misclassified_counts = np.zeros(10)
    
    for adv_image in adversarial_images:
        output = model(adv_image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        misclassified_counts[predicted_class] += 1
    
    # Find the class with the highest misclassification count
    most_misclassified_class = np.argmax(misclassified_counts)
    return most_misclassified_class

def main():
    attack_params = {
        "batch_size": 1,
        "epsilon": 0.1,
        "learning_rate": 0.01,
        "model_name": "model.pth"
    }

    model = ANNClassifier(attack_params)
    model.load_state_dict(torch.load(attack_params["model_name"]))
    model.eval()

    X_test = np.load("test data/X_test.npy", allow_pickle=True)
    y_test = np.load("test data/y_test.npy", allow_pickle=True).astype(int)

    X_selected, y_selected = generate_random_samples(X_test, y_test, num_samples_per_digit=3)

    X_selected = X_selected.view(-1, 1, 28, 28)

    attack = FGSM(model, torch.nn.CrossEntropyLoss(), epsilon=attack_params["epsilon"])

    results: dict = attack.apply(X_selected, y_selected)
    
    # check if the results contains a key named "adv_examples" and check each element
    # is a torch.Tensor. THIS IS IMPORTANT TO PASS THE TEST CASES!!!
    assert "adv_examples" in results.keys(), "Results should contain a key named 'adv_examples'"
    assert all([isinstance(x, torch.Tensor) for x in results["adv_examples"]]), "All elements in 'adv_examples' should be torch.Tensor"

    # check the image size should be 1x28x28
    assert results["adv_examples"][0].shape[1] == 1, "The image should be grayscale"
    assert results["adv_examples"][0].shape[2] == 28, "The image should be 28x28"
    
    print(f'Evasion Rate: {results["evasion_rate"]*100}%')

    plot_adversarial_examples(X_selected, results["adv_examples"])
    
    some_random_digit = random.randint(0, 9)
    most_misclassified_class = find_most_misclassified_digit(model, attack, some_random_digit, X_test=X_test, y_test=y_test)
    print(f"Most misclassified class for digit {some_random_digit} is: {most_misclassified_class}")

if __name__ == "__main__":
    main()
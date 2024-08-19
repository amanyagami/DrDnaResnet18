from src.models.resnet import resnet18
from random import randint
import random
import torchvision
import torchvision.transforms as transforms
from src.utils.helpers import device
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
activations= {}
def get_activation(name):
    def hook(model,input,output):
        activations[name] = output.detach()
    return hook

def listtohistogram(histolist):
    counts, bin_edges, _ = plt.hist(histolist, bins=10, edgecolor='black',density=True)
    # Create dictionary to store histogram data
    histogram_data = {}
    for i in range(len(bin_edges) - 1):
        bin_range = (bin_edges[i], bin_edges[i+1])
        histogram_data[bin_range] = counts[i]
    #print(histogram_data)
    return histogram_data


'''
tau processing here

'''
def tau1processing(tau1):
    tau1histtodict = {} 
    for neuron in tau1:
        tau1histtodict[neuron] = listtohistogram(tau1[neuron])
    return tau1histtodict
def tau2processing(tau2):
    tau2histtodict = {}
    for layer_name in tau2:
        tau2histtodict[layer_name] = listtohistogram(tau2[layer_name])
    return tau2histtodict
def tau3processing(tau3, count):
    tau3_activation_extremes = {}
    for layer_name in tau3:
        tau3[layer_name] /= count
        tensor= tau3[layer_name]
        flat_tensor = tensor.flatten()
        max_value, max_index = flat_tensor.max(0)
        min_value, min_index = flat_tensor.min(0)
        
        # Convert indices to tensor before using unravel_index
        max_index_tensor = torch.tensor(max_index.item())
        min_index_tensor = torch.tensor(min_index.item())
        
        # Get the index in the original tensor dimensions
        max_location = torch.unravel_index(max_index_tensor, tensor.shape)
        min_location = torch.unravel_index(min_index_tensor, tensor.shape)
        
        # Store the results in the output dictionary
        tau3_activation_extremes[layer_name] = {
            'max_value': max_value.item(),
            'max_location': max_location,
            'min_value': min_value.item(),
            'min_location': min_location
        }
    return tau3_activation_extremes


layer_names = [
    'conv1',
    'bn1',
    'layer1.0.conv1',
    'layer1.0.bn1',
    'layer1.0.conv2',
    'layer1.0.bn2',
    'layer2.0.conv1',
    'layer2.0.bn1',
    'layer2.0.conv2',
    'layer2.0.bn2',
    'layer3.0.conv1',
    'layer3.0.bn1',
    'layer3.0.conv2',
    'layer3.0.bn2',
    'layer4.0.conv1',
    'layer4.0.bn1',
    'layer4.0.conv2',
    'layer4.0.bn2',
    'avgpool',
    'fc'
]

# output dimension [ [] , []] -> out_dim (location of detection sites)
#cohort size = number of detection sites

layer_output_dims = {
    'conv1': [1, 64, 32, 32],
    'bn1': [1, 64, 32, 32],
    'layer1.0.conv1': [1, 64, 16, 16],
    'layer1.0.bn1': [1, 64, 16, 16],
    'layer1.0.conv2': [1, 64, 16, 16],
    'layer1.0.bn2': [1, 64, 16, 16],
    'layer2.0.conv1': [1, 128, 8, 8],
    'layer2.0.bn1': [1, 128, 8, 8],
    'layer2.0.conv2': [1, 128, 8, 8],
    'layer2.0.bn2': [1, 128, 8, 8],
    'layer3.0.conv1': [1, 256, 4, 4],
    'layer3.0.bn1': [1, 256, 4, 4],
    'layer3.0.conv2': [1, 256, 4, 4],
    'layer3.0.bn2': [1, 256, 4, 4],
    'layer4.0.conv1': [1, 512, 2, 2],
    'layer4.0.bn1': [1, 512, 2, 2],
    'layer4.0.conv2': [1, 512, 2, 2],
    'layer4.0.bn2': [1, 512, 2, 2],
    'avgpool': [1, 512, 1, 1],
    'fc': [1, 10]  # Assuming a final fully connected layer with 1000 outputs
}
def chooseRandomNeurons(layer_output_dims, cohort_size):
    final_list = []
    layer_names = layer_output_dims.keys()
    layer_names = list(layer_names)
    count = 0

    # Ensure that cohort_size is at least the number of layers
    if cohort_size < len(layer_output_dims):
        raise ValueError("Cohort size must be at least the number of layers")

    # Function to generate random neuron within a layer
    def generate_random_neuron(layer_name, dimensions):
        if layer_name == 'fc':
            height,width = dimensions
            h = random.randint(0, height - 1)
            w = random.randint(0, width - 1)
            return (layer_name, (h, w))
        else:
            _, channels, height, width = dimensions
            channel = random.randint(0, channels - 1)
            h = random.randint(0, height - 1)
            w = random.randint(0, width - 1)
            return (layer_name, (0, channel, h, w))
    
    # Step 1: Choose at least one neuron from each layer
    for layer_name in layer_names:
        dimensions = layer_output_dims[layer_name]
        neuron = generate_random_neuron(layer_name, dimensions)
        final_list.append(neuron)
        count += 1
    
    # Step 2: Choose remaining neurons randomly from any layer
    remaining_neurons = cohort_size - count
    while remaining_neurons > 0:
        layer_name = random.choice(layer_names)
        dimensions = layer_output_dims[layer_name]
        neuron = generate_random_neuron(layer_name, dimensions)
        final_list.append(neuron)
        remaining_neurons -= 1

    return final_list
cohort_size = 32

selected_neurons = chooseRandomNeurons(layer_output_dims, cohort_size)
with open('DetectionSites.txt', 'w') as f:
    for neuron in selected_neurons:
        layer_name, location = neuron
        f.write(f"{layer_name} ,{location}\n")
            

def offline_profiling(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    count = 0
    tau1 = {}
    tau2= {}
    tau3 = {}
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # single_input = inputs[0]  # Extract the first image in the batch
            # outputs = model(single_input.unsqueeze(0))  
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            
            #collection of data required for stats (tau 1, tau2, tau3)
            #tau 1 
            for neuron in selected_neurons:
                layer_name,pos = neuron
                #some layers dont have 4 dimenions so this check (eg Fc ->last layer resnet18)
                if len(pos) > 2:       
                    b,c,w,h=pos
                    if neuron in tau1:
                        tau1[neuron].append(activations[layer_name][b,c,w,h].item())
                    else:
                        tau1[neuron]= []
                        tau1[neuron].append(activations[layer_name][b,c,w,h].item())
                else :
                    w,h=pos
                    if neuron in tau1:
                        tau1[neuron].append(activations[layer_name][w,h].item())
                    else:
                        tau1[neuron]= []
                        tau1[neuron].append(activations[layer_name][w,h].item())  
            #Tau 2
            for layer_name, tensor in activations.items():
                # Flatten the tensor and convert to a list
                flat_list = tensor.flatten().tolist()
                # Store the 1D list in the output dictionary
                if layer_name in tau2:
                    tau2[layer_name] += flat_list
                else :
                    tau2[layer_name] = flat_list
            count += 1
            if count == 100:
                break
            
            
            #tau3
            for layer_name, tensor in activations.items():
                # Flatten the tensor to find max and min values and their indices
                
                if layer_name in tau3:
                    tau3[layer_name] += activations[layer_name]
                else:
                    tau3[layer_name] = activations[layer_name]
                    
        tau1 = tau1processing(tau1)
        tau2 = tau2processing(tau2)   
        tau3 = tau3processing(tau3,count)     
        print(tau2)            
        print(f'\nTest set: Average loss: {test_loss/len(testloader):.4f}, Accuracy: {correct}/{total} ({100.*correct/total:.2f}%)\n')    





   
if __name__ == '__main__':

    model = resnet18(pretrained=True, progress=True,device=device)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])


    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)



    #net.fc = nn.Linear(num_ftrs, 10)  # Modify the last layer to output 10 classes
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    num_ftrs = model.fc.in_features
    # Register hooks to all layers, including convolutional, batch normalization, and fully connected layers
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.bn1.register_forward_hook(get_activation('bn1'))
    model.layer1[0].conv1.register_forward_hook(get_activation('layer1.0.conv1'))
    model.layer1[0].bn1.register_forward_hook(get_activation('layer1.0.bn1'))
    model.layer1[0].conv2.register_forward_hook(get_activation('layer1.0.conv2'))
    model.layer1[0].bn2.register_forward_hook(get_activation('layer1.0.bn2'))
    model.layer2[0].conv1.register_forward_hook(get_activation('layer2.0.conv1'))
    model.layer2[0].bn1.register_forward_hook(get_activation('layer2.0.bn1'))
    model.layer2[0].conv2.register_forward_hook(get_activation('layer2.0.conv2'))
    model.layer2[0].bn2.register_forward_hook(get_activation('layer2.0.bn2'))
    model.layer3[0].conv1.register_forward_hook(get_activation('layer3.0.conv1'))
    model.layer3[0].bn1.register_forward_hook(get_activation('layer3.0.bn1'))
    model.layer3[0].conv2.register_forward_hook(get_activation('layer3.0.conv2'))
    model.layer3[0].bn2.register_forward_hook(get_activation('layer3.0.bn2'))
    model.layer4[0].conv1.register_forward_hook(get_activation('layer4.0.conv1'))
    model.layer4[0].bn1.register_forward_hook(get_activation('layer4.0.bn1'))
    model.layer4[0].conv2.register_forward_hook(get_activation('layer4.0.conv2'))
    model.layer4[0].bn2.register_forward_hook(get_activation('layer4.0.bn2'))
    model.avgpool.register_forward_hook(get_activation('avgpool'))
    model.fc.register_forward_hook(get_activation('fc'))

    offline_profiling(0)

    

   
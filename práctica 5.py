import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import DataLoader

# Asegúrate de actualizar esta ruta a la ubicación correcta en tu sistema de archivos
dataset_path = '/mnt/c/Users/Usuario/Desktop/facebook.tar'

# Cargar el dataset
dataset = FacebookDataset(root=dataset_path, name='ego-facebook')
data = dataset[0]

# Dividir el dataset en entrenamiento y prueba
train_dataset, test_dataset = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)

# Construir la GNN
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

# Configurar el entrenamiento
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Función de entrenamiento
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# Función de evaluación
def test(model, loader, criterion):
    model.eval()
    total_correct = 0
    for data in loader:
        out = model(data)
        pred = out[data.test_mask].max(dim=1)[1]
        total_correct += pred.eq(data.y[data.test_mask]).sum().item()
    return total_correct / len(loader.dataset)

# Entrenar y evaluar la GNN
for epoch in range(200):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_acc = test(model, test_loader, criterion)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')

print('Entrenamiento completado.')

from app.imports import *
from tqdm import tqdm


def fit_epoch(model, train_loader, criterion, optimizer):
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    all_preds = []
    all_labels = []

    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outp = model(inputs)
        loss = criterion(outp, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outp, 1)

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        running_corrects += torch.sum(preds == labels.data) 
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    train_f1 = f1_score(all_labels, all_preds, average='macro')
    return train_loss, train_acc, train_f1

def eval_epoch(model, val_loader, criterion):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    all_preds = []
    all_labels = []


    for inputs, labels in val_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            processed_size += inputs.size(0)
            
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    return val_loss, val_acc, val_f1

CHECKPOINT_PATH = "../checkpoints"


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int = 10,
    device: str = DEVICE,
    model_name: str = 'base_model'
) -> tuple:
    """
    Возвращает:
        history: словарь с историей метрик
        model: обученная модель
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    model = model.to(device)
    
    # Настройка цветов для tqdm
    COLORS = {
        'train': '\033[34m',  # синий
        'val': '\033[32m',    # зелёный
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    best_f1 = 0.0
    
    
    epoch_pbar = tqdm(range(epochs), desc='Epochs', position=0, leave=True)
    
    for epoch in epoch_pbar:
        # Обучение
        train_loss, train_acc, train_f1 = fit_epoch(model, train_loader, criterion, optimizer)
        # Валидация
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion)
        
        # Сохраняем метрики
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        

        # Сохраняем модель если улучшился F1 на валидации
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(CHECKPOINT_PATH, f'{model_name}.pt'))
            
            tqdm.write(f"Val F1: {val_f1:.4f} (saved)")
        
        # Выводим метрики.
        tqdm.write(
            f"Epoch {epoch + 1}/{epochs} | "
            f"{COLORS['bold']}Train{COLORS['end']}: "
            f"loss {COLORS['train']}{train_loss:.4f}{COLORS['end']} "
            f"acc {train_acc:.4f} "
            f"f1 {train_f1:.4f} | "
            f"{COLORS['bold']}Val{COLORS['end']}: "
            f"loss {COLORS['val']}{val_loss:.4f}{COLORS['end']} "
            f"acc {val_acc:.4f} "
            f"f1 {val_f1:.4f}"
        )
    
    return history, model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import re
from torch.utils.data import Dataset


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps, step):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    warmup_steps = float(learning_rate_warmup_steps)

    learning_rate *= (hidden_size ** -0.5)
    # Apply linear warmup
    learning_rate *= min(1.0, step / warmup_steps)
    # Apply rsqrt decay
    learning_rate *= (max(step, warmup_steps) ** -0.5)

    return learning_rate


def get_train_op_and_metrics(model, loss, params, step):
    """Generate training op and metrics."""
    learning_rate = get_learning_rate(
        learning_rate=params['learning_rate'],
        hidden_size=params['hidden_size'],
        learning_rate_warmup_steps=params['learning_rate_warmup_steps'],
        step=step
    )

    # Create optimizer, Adam optimizer with given parameters
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           betas=(params['optimizer_adam_beta1'], params['optimizer_adam_beta2']),
                           eps=params['optimizer_adam_epsilon'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_metrics = {'learning_rate': learning_rate, 'global_step': step}
    return optimizer, train_metrics


def record_scalars(metric_dict):
    """Print scalar metrics."""
    for key, value in metric_dict.items():
        print('records_scalars', key)
        if key == 'accuracy':
            print(f'{key}: {value}')
        else:
            print(f'{key}: {value}')


def get_assignment_map_from_checkpoint(model, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict((name, param) for name, param in model.named_parameters())

    checkpoint = torch.load(init_checkpoint, map_location=torch.device('cpu'))

    for name, param in checkpoint.items():
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1

    model.load_state_dict({name: checkpoint[name] for name in assignment_map}, strict=False)

    return assignment_map, initialized_variable_names


class MyDataset(Dataset):
    """Custom Dataset class for PyTorch."""

    def __init__(self, fname):
        self.data = self.parse_file(fname)

    def __len__(self):
        return len(self.data['inputs'])

    def __getitem__(self, idx):
        return {
            'inputs': self.data['inputs'][idx],
            'target_correct': self.data['target_correct'][idx],
            'target_id': self.data['target_id'][idx],
            'ids': self.data["ids"][idx],
            'correct': self.data["correct"][idx],
            'seq_len': self.data['seq_len'][idx]
        }

    def parse_file(self, fname):
        dataset = torch.load(fname)
        parsed_sample = parse_exmp(dataset)

        return parsed_sample


def parse_exmp(serial_exmp):
    """Convert a parsed example to a dictionary of tensors."""
    # 将稀疏张量转换为稠密张量
    inputs = serial_exmp['inputs'].to_dense()
    target_correct = serial_exmp['target_correct'].to_dense()
    target_id = serial_exmp['target_id'].to_dense()
    correct = serial_exmp['correct'].to_dense()
    ids = serial_exmp['ids'].to_dense()
    seq_len = serial_exmp['seq_len']

    # 转换数据类型
    inputs = inputs.int()
    target_correct = target_correct.float()
    target_id = target_id.int()
    correct = correct.float()
    ids = ids.int()
    seq_len = seq_len.int()

    # return [inputs, target_correct, target_id, ids, correct, seq_len]
    return {
        'inputs': inputs,
        'target_correct': target_correct,
        'target_id': target_id,
        'ids': ids,
        'correct': correct,
        'seq_len': seq_len
    }


def get_dataset(fname):
    """Load dataset and return as a PyTorch Dataset object."""
    return MyDataset(fname)


def get_padding(x, padding_value=0):
    """
    Args:
        x: int tensor with any shape
        padding_value: int value which padding value set
    Returns:
        float tensor with the same shape as x containing value 0,1
        0 means non-padding, 1 means padding
    """
    return (x == padding_value).float()


def get_padding_bias(x):
    """
    Calculate bias tensor from padding values in tensor
    Args:
        x: int tensor with shape [batch_size, length]
    Returns:
        Attention bias tensor of shape [batch_size, 1, 1, length]
    """
    padding = get_padding(x)
    attention_bias = padding * -1e9
    return attention_bias.unsqueeze(1).unsqueeze(1)

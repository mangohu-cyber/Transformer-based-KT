import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def dkt_loss(logits, target_correct, target_ids, seq_steps):
    """
    Calculate cross-entropy loss for DKT (Deep Knowledge Tracing)
    Args:
        logits: logits (no sigmoid) with shape [batch_size, length, vocab_size]
        target_correct: targets with shape [batch_size, length]
        target_ids: indices of target questions with shape [batch_size, length]
        seq_steps: lengths of sequences with shape [batch_size]

    Returns:
        cross-entropy loss
    """
    batch_size, length, vocab_size = logits.size()
    flat_logits = logits.view(-1, vocab_size)
    flat_target_correct = target_correct.view(-1)
    device = logits.device  # 获取 logits 所在的设备

    flat_base_target_index = torch.arange(batch_size * length) * vocab_size
    flat_bias_target_id = target_ids.view(-1)
    flat_target_id = flat_base_target_index.to(device) + flat_bias_target_id.to(device)

    flat_target_logits = flat_logits.view(-1)[flat_target_id]
    mask = torch.arange(length).expand(length).to(device) < seq_steps.unsqueeze(1).to(device)
    mask = mask.view(-1)

    flat_target_correct_mask = flat_target_correct[mask]
    flat_target_logits_mask = flat_target_logits[mask]

    loss = F.binary_cross_entropy_with_logits(
        flat_target_logits_mask,
        flat_target_correct_mask.float(),
        reduction='mean'
    )
    return loss


def calculate_acc(logits, target, target_ids, seq_len):
    """
    Percentage of times that predictions match labels on non-0s
    Args:
        logits: Tensor with shape [batch_size, length, vocab_size]
        target: Tensor with shape [batch_size, length]
        target_ids: indices of target questions with shape [batch_size, length]
        seq_len: lengths of sequences with shape [batch_size]

    Returns:
        Tensor representing the accuracy
    """
    batch_size, length, vocab_size = logits.size()
    device = logits.device

    # Flatten the logits and targets
    flat_logits = logits.view(-1, vocab_size)
    flat_target_correct = target.view(-1)
    flat_bias_target_id = target_ids.view(-1)

    # Calculate the flat index to select correct logits
    flat_base_target_index = torch.arange(batch_size * length) * vocab_size
    flat_target_id = flat_base_target_index.to(device) + flat_bias_target_id.to(device)

    # Select logits corresponding to target_ids
    flat_target_logits = flat_logits.view(-1)[flat_target_id]
    pred = torch.sigmoid(flat_target_logits.view(batch_size, length))

    binary_pred = (pred > 0.5).float()
    predict = binary_pred.view(-1)

    # Create a mask based on seq_len
    mask = torch.arange(length).expand(length).to(device) < seq_len.unsqueeze(1).to(device)
    mask = mask.view(-1)

    # Apply the mask
    flat_predict_mask = predict[mask]
    flat_target_mask = flat_target_correct[mask]

    # Calculate accuracy
    acc = (flat_target_mask == flat_predict_mask).float().mean()

    return acc


def calculate_auc(logits, target, target_ids, seq_len):
    """
    Calculate AUC for the predictions.
    Args:
        logits: Tensor with shape [batch_size, length, vocab_size]
        target: Tensor with shape [batch_size, length]
        target_ids: indices of target questions with shape [batch_size, length]
        seq_len: lengths of sequences with shape [batch_size]

    Returns:
        AUC score as a float
    """
    batch_size, length, vocab_size = logits.size()
    device = logits.device

    # Flatten the logits and targets
    flat_logits = logits.view(batch_size * length, vocab_size)
    flat_target_ids = target_ids.view(batch_size * length)
    flat_target = target.view(batch_size * length)

    # Select logits corresponding to target_ids using gather
    flat_target_logits = flat_logits.gather(1, flat_target_ids.unsqueeze(1).long()).squeeze(1)
    pred = torch.sigmoid(flat_target_logits)

    # Create a mask based on seq_len
    mask = torch.arange(length, device=device).expand(length) < seq_len.unsqueeze(1)

    # Apply the mask
    masked_pred = pred[mask.view(-1)]
    masked_target = flat_target[mask.view(-1)]

    # Check if mask is empty
    if masked_pred.numel() == 0 or masked_target.numel() == 0:
        return 0.0  # or raise an error

    # Move tensors to CPU and convert to numpy arrays for sklearn
    logits_np = masked_pred.detach().cpu().numpy()
    target_np = masked_target.detach().cpu().numpy()

    # Compute AUC
    auc = torch.tensor(roc_auc_score(target_np, logits_np), device=device)

    return auc


def calculate_doa(logits, target_correct, target_ids, seq_steps):
    # 将logits应用sigmoid得到预测正确概率
    probs = torch.sigmoid(logits)

    # 初始化DOA总和和计数器
    doa_sum = 0.0
    count = 0

    # 遍历所有学生组合
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            for k in range(seq_steps[i]):
                # 获取第i和第j个学生在第k个step的target_id
                id_i = target_ids[i, k]
                id_j = target_ids[j, k]

                if id_i == id_j:  # 确保两者的知识点是相同的
                    # 获取第i和第j个学生在第k个step的logits和target_correct
                    prob_i = probs[i, k]
                    prob_j = probs[j, k]
                    correct_i = target_correct[i, k]
                    correct_j = target_correct[j, k]

                    # 判断谁的知识状态更好
                    if prob_i > prob_j and correct_i > correct_j:
                        doa_sum += 1
                    elif prob_i < prob_j and correct_i < correct_j:
                        doa_sum += 1
                    elif prob_i == prob_j and correct_i == correct_j:
                        doa_sum += 1

                    count += 1

    # 计算DOA平均值
    doa = doa_sum / count if count > 0 else 0.0
    return doa

def _convert_to_eval_metric(metric_fn):
    """
    Wrapper for a metric function that returns scores as an eval metric function.
    The input metric_fn returns values for the current batch.
    The wrapper aggregates the return values collected over all of the batches evaluated.
    Args:
        metric_fn: function that returns scores for current batch's logits and targets

    Returns:
        function that aggregates the score from metric_fn
    """

    def problem_metric_fn(*args):
        """
        Return an aggregation of the metric_fn's returned values
        Args:
            *args: arguments to pass to the metric_fn

        Returns:
            Aggregated score as a scalar
        """
        score = metric_fn(*args)
        return score.mean()

    return problem_metric_fn


def get_eval_metrics(logits, labels, target_ids, seq_len):
    """
    Return a dictionary of model evaluation metrics
    Args:
        logits: output logits from the model
        labels: ground truth labels
        target_ids: indices of target questions
        seq_len: lengths of sequences

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'acc': _convert_to_eval_metric(calculate_acc)(logits, labels, target_ids, seq_len),
        'auc': _convert_to_eval_metric(calculate_auc)(logits, labels, target_ids, seq_len)

    }
    return metrics

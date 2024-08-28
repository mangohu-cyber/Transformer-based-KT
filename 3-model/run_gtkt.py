import time
from datetime import datetime

import numpy as np
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from model import DKT
from metrics import *
from utils import *
from model_params import *


# 参数设置
class Config:
    train_data = '../1-data/data_processed/assist09/train.pt'
    eval_data = '../1-data/data_processed/assist09/eval.pt'
    output_dir = '../4-output_dir'
    saved_model = os.path.join(output_dir, 'saved_model')
    eval_results = os.path.join(output_dir, 'eval_results.txt')
    results_graph = os.path.join(output_dir, 'results_graph.png')
    do_train = True
    do_eval = True
    do_predict = False


config = Config()


def build_dataloader(datadir, epoch, batch_size):
    dataset = MyDataset(datadir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader


# 训练和评估函数
def train(model, dataloader, optimizer, epoch, device):
    model.train()
    start_time = time.time()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        inputs = batch["inputs"].to(device)
        target_correct = batch["target_correct"].to(device)
        target_id = batch["target_id"].to(device)
        seq_steps = batch["seq_len"].to(device)

        optimizer.zero_grad()
        logits = model(inputs, target_id)
        loss = dkt_loss(logits, target_correct, target_id, seq_steps)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 32 == 0:
            end_time = time.time()
            cost_time = end_time - start_time
            print(
                f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                f'\tTrain Epoch: {epoch}'
                f'\tcurrent/total:[{batch_idx * len(inputs)}/{len(dataloader.dataset)}]'
                f'\tLoss: {loss.item():.5f}'
                f'\tcost_time:{cost_time:.2f}s'
            )

    result_eval_loss = total_loss / len(dataloader)

    return result_eval_loss


def evaluate(model, dataloader, device):
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs'].to(device)
            target_ids = batch['target_id'].to(device)
            target_correct = batch['target_correct'].to(device)
            seq_steps = batch['seq_len'].to(device)

            logits = model(inputs, target_ids)
            eval_loss += dkt_loss(logits, target_correct, target_ids, seq_steps).item()
            eval_metrics = get_eval_metrics(logits, target_correct, target_ids, seq_steps)

    result_eval_loss = eval_loss / len(dataloader)
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\t' +
          f'Eval set: Average loss: {result_eval_loss:.5f}\t' +
          ''.join([f'\t{k}: {v:.5f}' for k, v in eval_metrics.items()]))

    return result_eval_loss, eval_metrics


def save_results_graph(acc_np, auc_np, train_loss, eval_loss, graph):
    # 创建图形和两个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    epochs = range(1, len(acc_np) + 1)

    # 绘制训练和验证损失图
    ax1.plot(epochs, train_loss, 'bo-', label='Train')
    ax1.plot(epochs, eval_loss, 'r*-', label='Eval')
    ax1.set_title('Training and Evaluation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # 绘制训练和验证准确率图
    ax2.plot(epochs, acc_np, 'g^-', label='ACC')
    ax2.set_title('Evaluation ACC')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('ACC')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    # 绘制训练和验证准确率图
    ax3.plot(epochs, auc_np, 'y^-', label='AUC')
    ax3.set_title('Evaluation AUC')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('AUC')
    ax3.legend()
    ax3.grid(True, alpha=0.5)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图形
    plt.savefig(graph)

    # 显示图形
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DKT(BASE_PARAMS).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=BASE_PARAMS['learning_rate'],
        weight_decay=BASE_PARAMS['weight_decay'],
        betas=(BASE_PARAMS['optimizer_adam_beta1'], BASE_PARAMS['optimizer_adam_beta2']),
        eps=BASE_PARAMS['optimizer_adam_epsilon']
    )

    # clear eval results file
    if config.do_eval:
        with open(config.eval_results, "w") as writer:
            writer.write("START\n"
                         "metrics\t value\n")
            for key, value in BASE_PARAMS.items():
                writer.write(f"{key} = {value}\n")

    if config.do_train:
        train_loader = build_dataloader(config.train_data, BASE_PARAMS['epoch'], BASE_PARAMS['batch_size'])
        acc_np, auc_np, train_loss_np, eval_loss_np = [], [], [], []

        for epoch in range(1, BASE_PARAMS['epoch'] + 1):
            train_loss = train(model, train_loader, optimizer, epoch, device)

            if config.do_eval:
                eval_loader = build_dataloader(config.eval_data, 1, BASE_PARAMS['batch_size'])
                eval_loss, eval_metrics = evaluate(model, eval_loader, device)
                # 调参用的acc和loss
                acc_np.append(eval_metrics['acc'].cpu().numpy())
                auc_np.append(eval_metrics['auc'].cpu().numpy())
                train_loss_np.append(train_loss)
                eval_loss_np.append(eval_loss)

                # 将acc和loss保存为图
                save_results_graph(acc_np, auc_np, train_loss_np, eval_loss_np, config.results_graph)
                # 将每epoch的eval结果追加存入output_eval_file
                with open(config.eval_results, "a") as writer:
                    writer.write(f"\nEpoch: {epoch}\n"
                                 f"Eval loss: {eval_loss}\n")
                    for key, value in eval_metrics.items():
                        writer.write(f"{key} = {value}\n")
                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}' + f"\tEval saved to {config.eval_results}")

            # 按epoch保存模型
            model_save_path = os.path.join(config.saved_model, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}' + f"\tModel saved to {model_save_path}")

    if config.do_eval:
        eval_loader = build_dataloader(config.eval_data, 1, BASE_PARAMS['batch_size'])
        eval_loss, eval_metrics = evaluate(model, eval_loader, device)
        with open(config.eval_results, "a") as writer:
            writer.write(f"Final Eval loss: {eval_loss}\n")
            for key, value in eval_metrics.items():
                writer.write(f"{key} = {value}\n")

        print(f"\nEval saved to {config.eval_results}")
        # 保存最终模型
        final_model_save_path = os.path.join(config.saved_model, "final_model.pt")
        torch.save(model.state_dict(), final_model_save_path)
        print(f"Final model saved to {final_model_save_path}\n")


if __name__ == '__main__':
    main()

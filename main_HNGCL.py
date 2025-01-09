import logging
import pickle as pkl
from utility.dataloader import Data
from utility.return_meta import return_meta
from utility.model_logging_utils import get_next_log_filename, configure_logging
import time
import torch
import torch.optim as optim
import os
import utility.parser
import utility.batch_test
from model.HNGCL import HNGCL
import warnings


warnings.filterwarnings('ignore')


def main():
    utility.batch_test.set_seed(2023)
    # step 1: Check device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"
    args.device = device
    # step 2: Load data
    g_file = open(os.path.join(args.data_path + args.dataset, args.dataset + "_hg.pkl"), "rb")
    g = pkl.load(g_file)
    g_file.close()
    g = g.to(device)
    dataset = Data(args.data_path + args.dataset)
    print("Data loaded.")
    meta_paths, user_key, item_key, ui_relation = return_meta(args.dataset)
    args.meta_path_patterns = meta_paths
    args.user_key = user_key
    args.item_key = item_key
    args.ui_relation = ui_relation
    # step 3: Create model and training components
    model = HNGCL(g, args)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("Model created.")

    # step 4: Training
    print("Start training.")
    best_report_recall = 0.
    best_report_ndcg = 0.
    best_report_epoch = 0
    early_stop = 0
    for epoch in range(args.epochs):
        since = time.time()
        # Training and validation using a full graph
        if 1:
            result = utility.batch_test.Test(dataset, model, device, eval(args.topK), args.multicore,
                                             args.test_batch_size, long_tail=False)
            if result['recall'][0] > best_report_recall:
                early_stop = 0
                best_report_epoch = epoch + 1
                best_report_recall = result['recall'][0]
                best_report_ndcg = result['ndcg'][0]
                # print(args.next_log_filename)
                # torch.save(model.state_dict(), args.model + "_" + args.dataset)
            else:
                early_stop += 1

            if early_stop >= 20:
                print("early stop! best epoch:", best_report_epoch, "best_recall:", best_report_recall, ',best_ndcg:',
                      best_report_ndcg)
                logging.info(
                    f"best epoch: {best_report_epoch}, "
                    f"bset_recall: {best_report_recall:.5}, "
                    f"best_ndcg: {best_report_ndcg:.5} "
                )
                with open('./result/' + args.dataset + "/result.txt", "a") as f:
                    f.write(str(best_report_epoch) + " ")
                    f.write(str(best_report_recall) + " ")
                    f.write(str(best_report_ndcg) + "\n")
                break
            else:
                print("recall:", result['recall'], ",precision:", result['precision'], ',ndcg:', result['ndcg'])
                logging.info(
                    f"current epoch: {epoch + 1}, "
                    f"test_recall: {result['recall'][0]:.5}, "
                    f"test_ndcg: {result['ndcg'][0]:.5} "
                )

        model.train()
        sample_data = dataset.sample_data_to_train_all()
        users = torch.Tensor(sample_data[:, 0]).long()
        pos_items = torch.Tensor(sample_data[:, 1]).long()
        neg_items = torch.Tensor(sample_data[:, 2]).long()

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        users, pos_items, neg_items = utility.batch_test.shuffle(users, pos_items, neg_items)
        num_batch = len(users) // args.batch_size + 1
        average_loss = 0.
        average_reg_loss = 0.

        for batch_i, (batch_users, batch_positive, batch_negative) in enumerate(
                utility.batch_test.mini_batch(users, pos_items, neg_items, batch_size=args.batch_size)):
            batch_mf_loss, batch_emb_loss = model.bpr_loss(batch_users, batch_positive, batch_negative)
            batch_emb_loss = eval(args.regs)[0] * batch_emb_loss
            batch_loss = batch_emb_loss + batch_mf_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            average_loss += batch_mf_loss.item()
            average_reg_loss += batch_emb_loss.item()

        average_loss = average_loss / num_batch
        average_reg_loss = average_reg_loss / num_batch
        time_elapsed = time.time() - since
        print("\t Epoch: %4d| train time: %.3f | train_loss:%.4f + %.4f" % (
            epoch + 1, time_elapsed, average_loss, average_reg_loss))

    print("best epoch:", best_report_epoch)
    print("best recall:", best_report_recall)
    print("best ndcg:", best_report_ndcg)



if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = utility.parser.parse_args()
    args.model = "HNGCL"
    log_folder = args.model + "_log"  # 日志文件夹名称
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    next_log_filename = get_next_log_filename(log_folder)
    args.next_log_filename = next_log_filename
    configure_logging(next_log_filename)
    print(args)  # 在终端输出并记录到当前日志文件
    logging.info(args)
    main()

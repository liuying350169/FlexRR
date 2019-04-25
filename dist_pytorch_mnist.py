from __future__ import division, print_function

import argparse

import torch
import time
import numpy as np
import tqdm
import torch.nn.functional as F
from torch import distributed, nn
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

class Average(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count

    def __str__(self):
        return '{:.6f}'.format(self.average)


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, label):
        predictions = output.data.argmax(dim=1)
        correct = predictions.eq(label.data).sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)


class Trainer(object):
    #default
    # global parameter
    world_size = 3
    line = [20000, 40000]
    time = [-1]*world_size
    fast_worker_list= []
    howmanytrans = 0

    def __init__(self, net, optimizer, device, args):
        self.net = net
        self.optimizer = optimizer
        # self.train_loader = train_loader
        # self.test_loader = test_loader
        self.device = device
        self.args = args

    def get_dataloader(self, root, batch_size, rank, line):
        # rank is for which work
        # line is how many offset
        # line is [] for  p1 line0  p2 line1 p3
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        train_set = datasets.MNIST(
            root, train=True, transform=transform, download=True)
        #sampler = DistributedSampler(train_set)
        # train_loader = data.DataLoader(train_set,batch_size=batch_size,shuffle=(sampler is None),sampler=sampler)
        idxs = range(len(train_set))
        #default
        idxs_train = idxs
        if (rank == 0):
            idxs_train = idxs[:line[0]]
        if (rank == 1):
            idxs_train = idxs[line[0]:line[1]]
        if (rank == 2):
            idxs_train = idxs[line[1]:]

        train_loader = DataLoader(DatasetSplit(train_set, idxs_train), batch_size=batch_size, shuffle=True)

        test_loader = data.DataLoader(datasets.MNIST(root, train=False, transform=transform, download=True),
                                      batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def get_new_data_loader(self, root, batch_size, rank, line, task_to_trans, task_ready, op):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        train_set = datasets.MNIST(
            root, train=True, transform=transform, download=True)
        idxs = range(len(train_set))
        #default
        idxs_train = idxs
        if (rank == 0):
            idxs_train = idxs[:line[0]]
        if (rank == 1):
            idxs_train = idxs[line[0]:line[1]]
        if (rank == 2):
            idxs_train = idxs[line[1]:]

        if(op == 1):
        #trans to others
            task_ready += task_to_trans
            idxs_train -= task_ready
        elif(op == 0):
        #trans from others
            idxs_train += task_to_trans
            idxs_train -= task_ready

        train_loader = DataLoader(DatasetSplit(train_set, idxs_train), batch_size=batch_size, shuffle=True)

        test_loader = data.DataLoader(datasets.MNIST(root, train=False, transform=transform, download=True),
                                      batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def cal_line(self,time):
        #time is t0,t1,t2
        # t0 , line0, t1, line1 ,t2, line2, t3, line3, ... , ...
        t_sum = 0
        for t in range(len(time)):
            t_sum += time[t]
        t_avg = t_sum/len(time)

        for i in range(self.args.world_size-1):
            time_offset = time[i]-t_avg
            offset = abs(time_offset/t_avg)
            if i ==0:
                if(time_offset>=0):
                    #slow
                    self.line[i] = self.line[i]-(self.line[i] * offset)
                elif(time_offset<0):
                    #fast
                    self.line[i] = self.line[i]+(self.line[i] * offset)
            else:
                now_offset = self.line[i]-self.line[i-1]
                target_offset = self.line[i] * offset
                if(now_offset>target_offset):
                    #more
                    self.line[i] = self.line[i] - (now_offset-target_offset)
                elif(now_offset<target_offset):
                    #less
                    self.line[i] = self.line[i] + (target_offset-now_offset)
                else:
                    #bingo
                    break
        return self.line

    #more normal situation
    def fastworker_list(self):
        # self.time is receive after a epoch
        if len(self.time)< self.args.world_size/2:
            #top 1/2
            self.fast_worker_list.append(distributed.get_rank())

    def all_reduce_min(self, schedule, group):
        #tensor=,op=,group=
        distributed.all_reduce(schedule, op=distributed.ReduceOp.MIN, group=group)
        f = open('./print.txt', 'a')
        print("min",schedule[0])
        print("min", schedule[0], file=f)
        f.close()
        return schedule

    def all_reduce_max(self, schedule, group):
        # tensor=,op=,group=
        distributed.all_reduce(schedule, op=distributed.ReduceOp.MAX, group=group)
        f = open('./print.txt', 'a')
        print("max",schedule[0])
        print("max", schedule[0], file=f)
        f.close()
        return schedule

    #task dist
    def fit(self, epochs):
        #line = [20000,40000]
        time_begin = time.time()
        time_consume = 0
        for epoch in range(1, epochs + 1):
            #calculate line
            if(epoch>1):
                self.line = self.cal_line(self.time)

            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()
            time_end = time.time()
            time_consume = time_end - time_begin
            #calculate time consume t_rank
            #time[self.args.rank-1] = time_consume
            #self.fastworker_list()

            x = []
            if(self.time[distributed.get_rank()] == -1):
                distributed.recv(x,src=1)

            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc))

    def isstraggle(self,max,min):
        # is more than 20%    max-min/min
        max = max[0]
        min = min[0]
        if (max-min)/min >= 0.2:
            return True
        else:
            return False




    def train(self):
        #every epoch
        train_loss = Average()
        train_acc = Accuracy()
        self.net.train()
        train_loader, test_loader = self.get_dataloader(self.args.root, self.args.batch_size, self.args.rank, self.line)
        #new train_lodaer

        #new_group #ranks & timeout
        group_list = [i for i in range(self.args.world_size)]
        group = distributed.new_group(group_list)


        #needs global timeline
        counter = 0
        #default is 20000 sample batch_size is 1, max counter is 20000
        time_begin = time.time()
        total = len(train_loader)
        time_start = time.time()

        while(True):

            task_ready = []
            task_all = []
            for idx in enumerate(train_loader):
                task_all.append(idx)

            for idx, (data, label) in enumerate(train_loader):

                if self.args.rank == 0:
                    time.sleep(0)
                elif self.args.rank == 1:
                    time.sleep(0.4)
                elif self.args.rank == 2:
                    time.sleep(0.2)

                if(counter % 100 == 0):
                    #all_reduce
                    time_now = time.time()
                    time_consume = time_now - time_start
                    print("time_consume", time_consume)

                    schedule = torch.Tensor([time_consume])

                    if self.isstraggle(self.all_reduce_max(schedule=schedule, group=group),self.all_reduce_min(schedule=schedule, group=group)):
                    #if len(self.fast_worker_list)>0:
                        #calculate task_to_trans
                        target_worker = self.fast_worker_list[0]
                        percentage = counter / total
                        rest_task = total-counter
                        time_now = time.time()
                        time_consume = time_now - time_begin
                        time_predict = time_consume/percentage
                        target_worker_speed = total/self.time[target_worker]
                        local_worker_speed = total/time_predict
                        speed_avg = (target_worker_speed+local_worker_speed)/2
                        time_avg = rest_task/speed_avg
                        task_to_trans = rest_task - (time_avg*local_worker_speed)

                        rest_task_list = task_all - task_ready
                        task_to_trans_list = rest_task_list[:task_to_trans]

                        self.howmanytrans += task_to_trans
                        f = open('./print.txt','a')
                        print("howmanytrans",self.howmanytrans)
                        print("howmanytrans", self.howmanytrans, file=f)
                        f.close()
                        #send trans task
                        #x is part of rest_task
                        #local skip "task_to_trans" samples


                        #train_loader = self.get_dataloader(self.args.root, self.args.batch_size, self.args.rank, self.line)
                        #delete fast_worker from list
                        for i in range(len(self.fast_worker_list)-1):
                            self.fast_worker_list[i] = self.fast_worker_list[i+1]
                        #update time_consume in time
                        self.time[target_worker] = -1

                        #calculate new data_loader
                        if(schedule == self.all_reduce_min(schedule=schedule,group=group)):
                            op = 1
                            #send
                            target_worker = 0
                            distributed.isend(task_to_trans_list, dst=target_worker)

                            train_loader, test_loader = self.get_new_data_loader(self.args.root, self.args.batch_size,
                                                                    self.args.rank, self.line, task_to_trans_list, task_ready, op)
                            break
                        elif(schedule == self.all_reduce_max(schedule=schedule,group=group)):
                            op = 0
                            #recv
                            target_worker = 1
                            distributed.irecv(task_to_trans_list, dst=target_worker)

                            train_loader, test_loader = self.get_new_data_loader(self.args.root, self.args.batch_size,
                                                                    self.args.rank, self.line, task_to_trans_list, task_ready, op)
                            break

                data = data.to(self.device)
                label = label.to(self.device)

                output = self.net(data)
                loss = F.cross_entropy(output, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss.update(loss.item(), data.size(0))
                train_acc.update(output, label)
                counter += 1
                f = open('./print.txt', 'a')
                print("counter", counter)
                print("counter", counter, file=f)
                f.close()
                task_ready.append(idx)

            if(self.all_reduce_min(schedule=schedule,group=group) == 100):
                #the min is 100 means every worker is ready
                break
        return train_loss, train_acc

    def evaluate(self):
        test_loss = Average()
        test_acc = Accuracy()

        self.net.eval()

        with torch.no_grad():
            for data, label in self.test_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.net(data)
                loss = F.cross_entropy(output, label)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, label)

        return test_loss, test_acc


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        #error liuying
        #only integers, slices (`:`), ellipsis (`...`), None and long or byte Variables are valid indices (got numpy.float64)
        #print("item",item)
        #print("self.idxs[item]",self.idxs[item])
        image, label = self.dataset[int(self.idxs[item])]
        return image, label


def run(args):
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    net = Net().to(device)
    if use_cuda:
        net = nn.parallel.DistributedDataParallel(net)
    else:
        net = nn.parallel.DistributedDataParallelCPU(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    # train_loader, test_loader = get_dataloader(args.root, args.batch_size)

    trainer = Trainer(net, optimizer, device, args)

    trainer.fit(args.epochs)


def init_process(args):
    print(distributed.is_available())
    distributed.init_process_group(backend=args.backend, init_method=args.init_method, rank=args.rank, world_size=args.world_size)

    # distributed.init_process_group(
    #     backend=args.backend,
    #     init_method=args.init_method,
    #     rank=args.rank,
    #     world_size=args.world_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backend',
        type=str,
        default='gloo',
        help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument(
        '-r', '--rank', type=int, help='Rank of the current process.')
    parser.add_argument(
        '-s',
        '--world-size',
        type=int,
        help='Number of processes participating in the job.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    print(args)

    init_process(args)
    run(args)


if __name__ == '__main__':
    main()
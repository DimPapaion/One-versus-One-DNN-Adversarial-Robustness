from tqdm import tqdm
from Plots.Plots import *
import matplotlib.pyplot as plt
import gc
import datetime
import time

class BaseClassifier(object):
    def __init__(self, args,model,optimizer,criterion,scheduler,train_dl, valid_dl, test_dl, device, transform_matrix, state, model2=None):
        self.model = model
        self.model2 = model2
        self.optimizer = optimizer
        self.validation = args.validation
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_dl
        self.valid_loader = valid_dl
        self.test_loader = test_dl
        self.num_classes = args.n_classes
        self.epochs = args.epochs
        self.device = device
        self.model_name = args.model_name
        self.path = args.path_checkpoint
        self.transformOVO = transform_matrix
        self.target_type = args.target_type
        self.gamma = args.gamma
        self.schedule = args.schedule
        self.state = state
        self.dataset_name = args.dataset_name

    def reverse_OvO_labels(self, y, transform):
        reverse_trans = transform.T
        return [torch.sum(torch.from_numpy(reverse_trans) * y[i], axis=1) for i in range(len(y))]

    def accuracy(self, y_true, y_scores):
        labels, preds = 0, 0
        if self.target_type == 'OVO':
            y_true = self.reverse_OvO_labels(y=y_true, transform=self.transformOVO)
            y_scores = self.reverse_OvO_labels(y=y_scores, transform=self.transformOVO)
            y_true = torch.tensor(np.array([torch.Tensor.cpu(y_true[i]).detach().numpy() for i in range(len(y_true))]))
            y_scores = torch.tensor(np.array([torch.Tensor.cpu(y_scores[i]).detach().numpy() for i in range(len(y_scores))]))

            _, labels = torch.max(y_true, dim=1)

            _, preds = torch.max(y_scores, dim=1)
        else:
            # _, labels = torch.max(y_true, dim=1)

            _, preds = torch.max(y_scores, dim=1)

        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def plot_losses(self, tr_loss, val_loss):
        """ Plot the losses in each epoch"""

        plt.plot(tr_loss, '-bx')
        plt.plot(val_loss, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.show()

    def adjust_learning_rate(self, optimizer,epoch):

        if epoch in self.schedule:
            self.state['lr'] *= self.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.state['lr']

    def training(self):
        losst = 0.0
        acc = 0.0
        total = 0

        self.model = self.model.to(self.device)
        self.model.train(True)

        for data, target in tqdm(self.train_loader):
            target = target.float()
            data, target = data.to(self.device), target.to(self.device)


            self.optimizer.zero_grad()
            _, _, _,output = self.model(data)

            target = (target + 1) / 2
            output = (output + 1) / 2

            # print(output.shape, target.shape)

            # loss = self.criterion(output, output3, output4, target2.argmax(dim=1))
            loss = self.criterion(output, target)

            losst += loss.item()  # * data.size(0)
            total += target.size(0)
            # AP += self.AP_scores(torch.Tensor.cpu(target.long()).detach().numpy(), torch.Tensor.cpu(output).detach().numpy())
            acc += self.accuracy(torch.Tensor.cpu(target).detach(), torch.Tensor.cpu(output).detach())

            # loss.requires_grad = True
            loss.backward()
            self.optimizer.step()


            del data, target, output
            gc.collect()
            torch.cuda.empty_cache()

        num_batches = float(len(self.train_loader))
        total_loss_ = losst / num_batches
        total_acc_ = acc / num_batches
        return total_loss_, total_acc_

    def evaluation(self):

        self.model.eval()
        if self.model2 is not None:
            self.model2.eval()
        losst = 0.0
        acc = 0.0
        total = 0.0

        with torch.no_grad():
            for data, target in tqdm(self.valid_loader):
                target = target.float()
                data, target = data.to(self.device), target.to(self.device)
                _, _, _,output = self.model(data)

                target = (target + 1) / 2
                output = (output + 1) / 2

                # output = torch.tensor(output, dtype=torch.float64)

                loss = self.criterion(output, target)

                losst += loss.item()
                total += target.size(0)

                acc += self.accuracy(torch.Tensor.cpu(target).detach(), torch.Tensor.cpu(output).detach())

            num_batches = float(len(self.valid_loader))
            val_loss_ = losst / num_batches
            val_acc_ = acc / num_batches
        return val_loss_, val_acc_

    def train_model(self, validation=False):
        tr_loss, tr_acc = [], []
        val_loss, val_acc = [], []

        best_acc = 0.0

        start_time = time.time()
        for epoch in range(self.epochs):
            print("-------Epoch {}/{}---------".format(epoch + 1, self.epochs))

            if self.validation:
                if self.scheduler is not None:
                    self.scheduler.step()
                else:
                    self.adjust_learning_rate(optimizer=self.optimizer, epoch=epoch)
                    print('LR: %f' % (self.state['lr']))

                tr_loss_, tr_acc_ = self.training()
                val_loss_, val_acc_ = self.evaluation()


                tr_loss.append(tr_loss_), tr_acc.append(tr_acc_)
                val_loss.append(val_loss_), val_acc.append(val_acc_)

                print(
                    "Train Loss: {:.10f}, Train Accuracy: {:.6f}%, Validation Loss: {:.10f}, Validation Accuracy: {:.6f}%".format(
                        tr_loss_,
                        tr_acc_ * 100,
                        val_loss_,
                        val_acc_ * 100
                    ))

                # if epoch % 20 == 0:
                #     self.plot_losses(tr_loss, val_loss)

                if best_acc < val_acc_:
                    best_acc = val_acc_
                    checkpoint = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                  'optimizer_model': self.optimizer.state_dict()}
                    torch.save(checkpoint, './' + self.dataset_name +'_OVA_32x32_v1.tar')
                    print("-----Check Point in epoch {} saved  -----".format(int(epoch + 1)))

            else:
                tr_loss_, tr_acc_ = self.training()

                tr_loss.append(tr_loss_), tr_acc.append(tr_acc_)
                print("Train Loss: {:.6f}, Train Accuracy: {:.4f}% ".format(tr_loss_, tr_acc_ * 100))
                if best_acc < tr_acc_:
                    best_acc = tr_acc_
                    # if epoch +1 == self.epochs:

                    torch.save(self.model.state_dict(), self.path)
                    print("-----Check Point in epoch {} saved in path: {} -----".format(int(epoch + 1), str(self.path)))

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

        if self.validation:
            return ([tr_loss, tr_acc], [val_loss, val_acc])
        else:
            return ([tr_loss, tr_acc])

    def test_model(self, model2=None):
        self.model.to(self.device)
        if model2 is None:
            self.model.eval()
        else:
            self.model = model2
            self.model.to(self.device)
            self.model.eval()
        losst = 0.0
        acc = 0.0

        y_pred = np.empty((0, self.num_classes), float)
        y_true = np.empty((0, self.num_classes), float)

        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                # target = target.float()
                data, target = data.to(self.device), target.to(self.device)

                bs, c, h, w = data.size()
                _,_,_,output = self.model(data.view(-1, c, h, w))
                loss = self.criterion(output, target)

                losst += loss
                acc += self.accuracy(torch.Tensor.cpu(target).detach().numpy(),
                                     torch.Tensor.cpu(output).detach().numpy())

                # y_pred =0 np.append(y_pred, torch.Tensor.cpu(output).detach().numpy(), axis=0)
                # y_true = 0np.append(y_true, torch.Tensor.cpu(target).detach().numpy(), axis=0)

        num_samples = float(len(self.test_loader))
        test_loss = losst / num_samples
        test_acc = acc / num_samples

        print("test_loss: {:.6f}, test_Accuracy: {:.4f}%".format(test_loss, test_acc))
        return test_loss, test_acc, y_pred, y_true



class BaseClassifierOvA(object):
    def __init__(self, args,model,optimizer,criterion,scheduler,train_dl, valid_dl, test_dl, device, transform_matrix, state, model2=None):
        self.model = model
        self.model2 = model2
        self.optimizer = optimizer
        self.validation = args.validation
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_dl
        self.valid_loader = valid_dl
        self.test_loader = test_dl
        self.num_classes = args.n_classes
        self.epochs = args.epochs
        self.device = device
        self.model_name = args.model_name
        self.path = args.path_checkpoint
        self.transformOVO = transform_matrix
        self.target_type = args.target_type
        self.gamma = args.gamma
        self.schedule = args.schedule
        self.state = state
        self.dataset_name = args.dataset_name



    def accuracy(self, y_true, y_scores):

        _, preds = torch.max(y_scores, dim=1)

        return torch.tensor(torch.sum(preds == y_true).item() / len(preds))

    def plot_losses(self, tr_loss, val_loss):
        """ Plot the losses in each epoch"""

        plt.plot(tr_loss, '-bx')
        plt.plot(val_loss, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.show()

    def adjust_learning_rate(self, optimizer,epoch):

        if epoch in self.schedule:
            self.state['lr'] *= self.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.state['lr']

    def training(self):
        losst = 0.0
        acc = 0.0
        total = 0

        self.model = self.model.to(self.device)
        self.model.train(True)

        for data, target in tqdm(self.train_loader):

            data, target = data.to(self.device), target.to(self.device)


            self.optimizer.zero_grad()
            # _,_,_,output = self.model(data)
            _,_,_,output = self.model(data)
            # print(output.shape, target.shape)

            # loss = self.criterion(output, output3, output4, target2.argmax(dim=1))
            loss = self.criterion(output, target)

            losst += loss.item()  # * data.size(0)
            total += target.size(0)
            # AP += self.AP_scores(torch.Tensor.cpu(target.long()).detach().numpy(), torch.Tensor.cpu(output).detach().numpy())
            acc += self.accuracy(torch.Tensor.cpu(target).detach(), torch.Tensor.cpu(output).detach())

            # loss.requires_grad = True
            loss.backward()
            self.optimizer.step()


            del data, target, output
            gc.collect()
            torch.cuda.empty_cache()

        num_batches = float(len(self.train_loader))
        total_loss_ = losst / num_batches
        total_acc_ = acc / num_batches
        return total_loss_, total_acc_

    def evaluation(self):

        self.model.eval()
        if self.model2 is not None:
            self.model2.eval()
        losst = 0.0
        acc = 0.0
        total = 0.0

        with torch.no_grad():
            for data, target in tqdm(self.valid_loader):

                data, target = data.to(self.device), target.to(self.device)
                # _,_,_,output = self.model(data)
                _,_,_,output = self.model(data)



                # output = torch.tensor(output, dtype=torch.float64)

                loss = self.criterion(output, target)

                losst += loss.item()
                total += target.size(0)

                acc += self.accuracy(torch.Tensor.cpu(target).detach(), torch.Tensor.cpu(output).detach())

            num_batches = float(len(self.valid_loader))
            val_loss_ = losst / num_batches
            val_acc_ = acc / num_batches
        return val_loss_, val_acc_

    def train_model(self, validation=False):
        tr_loss, tr_acc = [], []
        val_loss, val_acc = [], []

        best_acc = 0.0

        start_time = time.time()
        for epoch in range(self.epochs):
            print("-------Epoch {}/{}---------".format(epoch + 1, self.epochs))

            if self.validation:
                if self.scheduler is not None:
                    self.scheduler.step()
                else:
                    self.adjust_learning_rate(optimizer=self.optimizer, epoch=epoch)
                    print('LR: %f' % (self.state['lr']))

                tr_loss_, tr_acc_ = self.training()
                val_loss_, val_acc_ = self.evaluation()


                tr_loss.append(tr_loss_), tr_acc.append(tr_acc_)
                val_loss.append(val_loss_), val_acc.append(val_acc_)

                print(
                    "Train Loss: {:.10f}, Train Accuracy: {:.6f}%, Validation Loss: {:.10f}, Validation Accuracy: {:.6f}%".format(
                        tr_loss_,
                        tr_acc_ * 100,
                        val_loss_,
                        val_acc_ * 100
                    ))

                # if epoch % 20 == 0:
                #     self.plot_losses(tr_loss, val_loss)

                if best_acc < val_acc_:
                    best_acc = val_acc_
                    checkpoint = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                  'optimizer_model': self.optimizer.state_dict()}
                    torch.save(checkpoint, './' + self.dataset_name +'_OVA_V6.tar')
                    print("-----Check Point in epoch {} saved  -----".format(int(epoch + 1)))

            else:
                tr_loss_, tr_acc_ = self.training()

                tr_loss.append(tr_loss_), tr_acc.append(tr_acc_)
                print("Train Loss: {:.6f}, Train Accuracy: {:.4f}% ".format(tr_loss_, tr_acc_ * 100))
                if best_acc < tr_acc_:
                    best_acc = tr_acc_
                    # if epoch +1 == self.epochs:

                    torch.save(self.model.state_dict(), self.path)
                    print("-----Check Point in epoch {} saved in path: {} -----".format(int(epoch + 1), str(self.path)))

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

        if self.validation:
            return ([tr_loss, tr_acc], [val_loss, val_acc])
        else:
            return ([tr_loss, tr_acc])

    def test_model(self, model2=None):
        self.model.to(self.device)
        if model2 is None:
            self.model.eval()
        else:
            self.model = model2
            self.model.to(self.device)
            self.model.eval()
        losst = 0.0
        acc = 0.0

        y_pred = np.empty((0, self.num_classes), float)
        y_true = np.empty((0, self.num_classes), float)

        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                # target = target.float()
                data, target = data.to(self.device), target.to(self.device)

                bs, c, h, w = data.size()
                _,_,_,output = self.model(data)
                loss = self.criterion(output, target)

                losst += loss
                acc += self.accuracy(torch.Tensor.cpu(target).detach().numpy(),
                                     torch.Tensor.cpu(output).detach().numpy())

                # y_pred =0 np.append(y_pred, torch.Tensor.cpu(output).detach().numpy(), axis=0)
                # y_true = 0np.append(y_true, torch.Tensor.cpu(target).detach().numpy(), axis=0)

        num_samples = float(len(self.test_loader))
        test_loss = losst / num_samples
        test_acc = acc / num_samples

        print("test_loss: {:.6f}, test_Accuracy: {:.4f}%".format(test_loss, test_acc))
        return test_loss, test_acc, y_pred, y_true








#
# def adjust_learning_rate(optimizer, epoch):
#     global state
#     if epoch in args.schedule:
#         state['lr'] *= args.gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = state['lr']


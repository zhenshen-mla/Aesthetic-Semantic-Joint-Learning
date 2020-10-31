import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_process.processing_531 import make_loader
from model_arch.network_resnet import make_network
from tensorboardX import SummaryWriter
from lr_scheduler import LR_Scheduler
from sklearn import metrics
import warnings

'''
:parameters
# '''
# def hook_fn_forward_fc(module, input, output):
#     print('fc forward  :', 'x=%.6f, y=%.6f' % (torch.norm(input[0], 2).item(), torch.norm(output[0], 2).item()), input[0].shape, output[0].shape)
# def hook_fn_backward_fc(module, grad_input, grad_output):
#     print('fc backward :', 'dy=%.6f, db=%.6f, dx=%.6f, dw=%.6f' % (torch.norm(grad_output[0], 2).item(), torch.norm(grad_input[0], 2).item(), torch.norm(grad_input[1], 2).item(), torch.norm(grad_input[2], 2).cpu().item()), grad_output[0].shape, grad_input[0].shape, grad_input[1].shape, grad_input[2].shape)
# def hook_fn_forward_conv(module, input, output):
#     print('conv forward  :', 'x=%.6f, y=%.6f' % (torch.norm(input[0], 2).item(), torch.norm(output[0], 2).item()),
#           input[0].shape, output[0].shape)
# def hook_fn_backward_conv(module, grad_input, grad_output):
#     print('conv backward :', 'dy=%.6f, dx=%.6f, dw=%.6f' % (
#     torch.norm(grad_output[0], 2).item(), torch.norm(grad_input[0], 2).item(), torch.norm(grad_input[1], 2).item()), grad_output[0].shape, grad_input[0].shape, grad_input[1].shape)

# 74.10 59.97 batch=16 lr=1e-4 epoch=50
ID = 0
COMMENT = '_1'
LR = 0.00001
EPOCHS = 200
LR_STEP = 25

# (1) distribute GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(ID)
# (2) tensorboardX
writer = SummaryWriter(comment=COMMENT)


def main():

    best_pred_seg = 0.0
    best_pred_aes = 0.0

    lr = LR
    num_epochs = EPOCHS

    print('\nloading the dataset ...\n')
    train_data, val_data, trainloader, valloader = make_loader()
    print(len(train_data), len(val_data), len(trainloader), len(valloader))
    print('done')

    print('\nloading the network ...\n')
    model = make_network()
    # for name, module in model.named_modules():
    #     if name == 'A_fc':
    #         print(name)
    #         module.register_forward_hook(hook_fn_forward_fc)
    #         module.register_backward_hook(hook_fn_backward_fc)
    #     if name == 'B_fc':
    #         print(name)
    #         module.register_forward_hook(hook_fn_forward_fc)
    #         module.register_backward_hook(hook_fn_backward_fc)
    #     if name == 'layer4.0.A_conv3':
    #         print(name)
    #         module.register_forward_hook(hook_fn_forward_conv)
    #         module.register_backward_hook(hook_fn_backward_conv)
    #     if name == 'layer4.0.B_conv3':
    #         print(name)
    #         module.register_forward_hook(hook_fn_forward_conv)
    #         module.register_backward_hook(hook_fn_backward_conv)
    #     if name == 'layer4.1.A_conv3':
    #         print(name)
    #         module.register_forward_hook(hook_fn_forward_conv)
    #         module.register_backward_hook(hook_fn_backward_conv)
    #     if name == 'layer4.1.B_conv3':
    #         print(name)
    #         module.register_forward_hook(hook_fn_forward_conv)
    #         module.register_backward_hook(hook_fn_backward_conv)
    #     if name == 'layer4.2.A_conv3':
    #         print(name)
    #         module.register_forward_hook(hook_fn_forward_conv)
    #         module.register_backward_hook(hook_fn_backward_conv)
    #     if name == 'layer4.2.B_conv3':
    #         print(name)
    #         module.register_forward_hook(hook_fn_forward_conv)
    #         module.register_backward_hook(hook_fn_backward_conv)

    criterion_aes = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss()

    ## move to GPU
    print('\nmoving to GPU ...\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion_aes.to(device)
    criterion_seg.to(device)

    ### optimizer

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
    scheduler = LR_Scheduler(mode='step', base_lr=lr, num_epochs=num_epochs, iters_per_epoch=len(trainloader), lr_step=LR_STEP)

    # training
    print('\nstart training ...\n')

    for epoch in range(num_epochs):

        running_loss_aes = 0.0
        running_correct_aes = 0
        running_total_aes = 0

        running_loss_seg = 0.0
        running_correct_seg = 0
        running_total_seg = 0

        model.train()
        for batch_idx, (data, target_aes, target_seg) in enumerate(trainloader):
            data, target_aes, target_seg = data.to(device), target_aes.to(device), target_seg.to(device)
            scheduler(optimizer, batch_idx, epoch, best_pred_aes, best_pred_seg)
            optimizer.zero_grad()

            # forward
            pred_aes, pred_seg = model(data)
            # backward
            loss_aes = criterion_aes(pred_aes, target_aes)
            loss_seg = criterion_seg(pred_seg, target_seg)
            # print('loss : ', loss_aes, loss_seg)
            loss = loss_aes + loss_seg

            loss.backward()
            optimizer.step()

            predict_aes = torch.argmax(pred_aes, 1)
            correct_aes = torch.eq(predict_aes, target_aes).sum().double().item()
            predict_seg = torch.argmax(pred_seg, 1)
            correct_seg = torch.eq(predict_seg, target_seg).sum().double().item()

            running_loss_aes += loss_aes.item()
            running_loss_seg += loss_seg.item()
            running_correct_aes += correct_aes
            running_correct_seg += correct_seg
            running_total_aes += target_aes.size(0)
            running_total_seg += target_seg.size(0)

        loss_aes = running_loss_aes * 32 / running_total_aes
        accuracy_aes = 100 * running_correct_aes / running_total_aes
        loss_seg = running_loss_seg * 32 / running_total_seg
        accuracy_seg = 100 * running_correct_seg / running_total_seg

        writer.add_scalar('scalar/loss_aes_train', loss_aes, epoch)
        writer.add_scalar('scalar/loss_seg_train', loss_seg, epoch)
        writer.add_scalar('scalar/accuracy_aes_train', accuracy_aes, epoch)
        writer.add_scalar('scalar/accuracy_seg_train', accuracy_seg, epoch)

        print('aes training ',
              '    Epoch[%d /50],loss = %.6f,accuracy=%.4f %%' %
              (epoch + 1, loss_aes, accuracy_aes))
        print('seg  training ',
              '    Epoch[%d /50],loss = %.6f,accuracy=%.4f %%' %
              (epoch + 1, loss_seg, accuracy_seg))
        print('previous best ',
              '    Epoch[%d /50], best_pred_aes=%.4f %%, best_pred_seg=%.4f %%' %
              (epoch + 1, best_pred_aes, best_pred_seg))

        model.eval()
        with torch.no_grad():
            running_loss_aes = 0.0
            running_correct_aes = 0
            running_total_aes = 0

            running_loss_seg = 0.0
            running_correct_seg = 0
            running_total_seg = 0

            for batch_idx, (data, target_aes, target_seg) in enumerate(valloader):
                data, target_aes, target_seg = data.to(device), target_aes.to(device), target_seg.to(device)
                optimizer.zero_grad()
                # forward
                pred_aes, pred_seg = model(data)
                # backward
                loss_aes = criterion_aes(pred_aes, target_aes)
                loss_seg = criterion_seg(pred_seg, target_seg)

                predict_aes = torch.argmax(pred_aes, 1)
                correct_aes = torch.eq(predict_aes, target_aes).sum().double().item()
                predict_seg = torch.argmax(pred_seg, 1)
                correct_seg = torch.eq(predict_seg, target_seg).sum().double().item()

                running_loss_aes += loss_aes.item()
                running_loss_seg += loss_seg.item()
                running_correct_aes += correct_aes
                running_correct_seg += correct_seg
                running_total_aes += target_aes.size(0)
                running_total_seg += target_seg.size(0)

            loss_aes = running_loss_aes * 32 / running_total_aes
            accuracy_aes = 100 * running_correct_aes / running_total_aes
            loss_seg = running_loss_seg * 32 / running_total_seg
            accuracy_seg = 100 * running_correct_seg / running_total_seg

            if accuracy_aes > best_pred_aes:
                best_pred_aes = accuracy_aes
            if accuracy_seg > best_pred_seg:
                best_pred_seg = accuracy_seg

            writer.add_scalar('scalar/loss_aes_val', loss_aes, epoch)
            writer.add_scalar('scalar/loss_seg_val', loss_seg, epoch)
            writer.add_scalar('scalar/accuracy_aes_val', accuracy_aes, epoch)
            writer.add_scalar('scalar/accuracy_seg_val', accuracy_seg, epoch)

            print('aes valing',
                  '    Epoch[%d /50],loss = %.6f,accuracy=%.4f %%' %
                  (epoch + 1, loss_aes, accuracy_aes))
            print('seg valing',
                  '    Epoch[%d /50],loss = %.6f,accuracy=%.4f %%' %
                  (epoch + 1, loss_seg, accuracy_seg))
#


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
    writer.close()
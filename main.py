import torch
from torchvision.transforms import transforms as T
import argparse
from torch import optim
import cv2
from dataset.dataset import ImageDataset, ImageDatasetNoTile
from torch.utils.data import DataLoader
from loss.focalloss import FocalLoss, EdgeAttBinaryCrossEntropyLoss, EdgeAttBinaryFocalLoss
from loss.diceloss import DiceLoss
from loss.relaxloss import RelaxLoss
from loss.gaussianloss import GaussianLoss
from torchmetrics import F1
from metric.hd import HD95
from network.unet import VanillaUNet, UNet_2Plus, AttentionUNet
from network.gscnn import GSCNN
from network.deeplabv3p import DeeplabV3Plus
from network.decoupleunet import DecoupleSegNet, DecoupleUNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.6978, 0.6504, 0.7477], [0.1347, 0.1612, 0.1641])
    # the mean value and standard deviation of our dataset have been calculated
])

y_transform = T.Compose([
    T.ToTensor()  # transform for ground truth
])


def train_model(model, optimizer, dataload, dataload_test, num_epochs):
    # set learning rate decliner
    decliner = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # set metrics
    metric_f1 = F1(num_classes=1)
    metric_hd95 = HD95()
    max_f1 = 0

    save_path="G:/python project/final/tt.pth"

    if args.model == "U-Net" or args.model == "U-Net++" or args.model == "Attention U-Net" or args.model == "DeeplabV3+":
        criterion_bce = FocalLoss(gamma=1, adaptive_weight=True)
        for epoch in range(num_epochs):
            model.train()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            dataset_size = len(dataload.dataset)
            epoch_loss = 0
            step = 0
            for x, y, _ in dataload:
                optimizer.zero_grad()
                inputs = x.to(device)
                masks = y.to(device)
                outputs = model(inputs)
                loss = criterion_bce(outputs, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                step += 1
                print("%d/%d,train_loss:%0.4f" % (step, dataset_size // dataload.batch_size, loss.item()))
            print("epoch %d loss:%0.4f" % (epoch, epoch_loss))
            decliner.step()
            model.eval()
            with torch.no_grad():
                for x, y, c in dataload_test:
                    masks = y.to(int)
                    masks = masks.squeeze()
                    masks = torch.flatten(masks, start_dim=0, end_dim=1)
                    x = x.to(device)
                    outputs_test = model(x)
                    output_test = outputs_test.to('cpu')
                    output_test_f1 = output_test.squeeze()
                    output_test_f1 = torch.flatten(output_test_f1, start_dim=0, end_dim=1)
                    f1 = metric_f1(output_test_f1, masks)
                    hd95 = metric_hd95(output_test, c)
            f1 = metric_f1.compute()
            hd95 = metric_hd95.compute()
            if f1 > max_f1:
                max_f1 = f1
                model_state_dict = model.state_dict()
                torch.save(model_state_dict, save_path)
            print('f1:%0.4f' % f1)
            print('hd95:%0.4f' % hd95)
            metric_f1.reset()
            metric_hd95.reset()
        print('max_f1:%0.4f' % max_f1)
    elif args.model == "GSCNN":
        criterion_edgeattbce = EdgeAttBinaryCrossEntropyLoss()
        criterion_gaussian = GaussianLoss()
        criterion_bce_1 = FocalLoss(gamma=1, alpha=0.9, adaptive_weight=True)
        criterion_bce_2 = FocalLoss(gamma=1, alpha=0.9, adaptive_weight=True)
        for epoch in range(num_epochs):
            model.train()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            dataset_size = len(dataload.dataset)
            epoch_loss = 0
            step = 0
            for x, y, c in dataload:
                optimizer.zero_grad()
                inputs = x.to(device)
                masks = y.to(device)
                contours = c.to(device)
                outputs = model(inputs)

                loss1 = criterion_bce_1(outputs[0], masks)
                loss2 = criterion_bce_2(outputs[1], contours)
                loss3 = criterion_gaussian(outputs[0], masks)
                loss4 = criterion_edgeattbce(outputs[0], masks, outputs[1])

                loss = loss1 + loss2 + 0.1 * loss3 + loss4

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                step += 1
                print("%d/%d,train_loss:%0.4f" % (step, dataset_size // dataload.batch_size, loss.item()))
            print("epoch %d loss:%0.4f" % (epoch, epoch_loss))
            decliner.step()
            model.eval()
            with torch.no_grad():
                for x, y, c in dataload_test:
                    masks = y.to(int)
                    masks = masks.squeeze()
                    masks = torch.flatten(masks, start_dim=0, end_dim=1)
                    x = x.to(device)
                    outputs_test = model(x)
                    output_test = outputs_test[0].to('cpu')
                    output_test_f1 = output_test.squeeze()
                    output_test_f1 = torch.flatten(output_test_f1, start_dim=0, end_dim=1)
                    f1 = metric_f1(output_test_f1, masks)
                    hd95 = metric_hd95(output_test, c)
            f1 = metric_f1.compute()
            hd95 = metric_hd95.compute()
            if f1 > max_f1:
                max_f1 = f1
                model_state_dict = model.state_dict()
                torch.save(model_state_dict, save_path)
            print('f1:%0.4f' % f1)
            print('hd95:%0.4f' % hd95)
            metric_f1.reset()
            metric_hd95.reset()
        print('max_f1:%0.4f' % max_f1)

    elif args.model == "DecoupleSegNet":
        criterion_relax = RelaxLoss(classes=2)
        criterion_edgeattfl = EdgeAttBinaryFocalLoss()
        criterion_bce_1 = FocalLoss(gamma=1, alpha=0.9, adaptive_weight=True)
        criterion_bce_2 = FocalLoss(gamma=1, alpha=0.9, adaptive_weight=True)
        for epoch in range(num_epochs):
            model.train()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            dataset_size = len(dataload.dataset)
            epoch_loss = 0
            step = 0
            for x, y, c in dataload:
                optimizer.zero_grad()
                inputs = x.to(device)
                masks = y.to(device)
                contours = c.to(device)
                outputs = model(inputs)

                loss1 = criterion_bce_1(outputs[0], masks)
                loss2 = criterion_bce_2(outputs[2], contours)
                loss3 = criterion_relax(outputs[1], masks)
                loss4 = criterion_edgeattfl(outputs[0], masks, outputs[2])

                loss = loss1 + loss2 + loss3 + loss4

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                step += 1
                print("%d/%d,train_loss:%0.4f" % (step, dataset_size // dataload.batch_size, loss.item()))
            print("epoch %d loss:%0.4f" % (epoch, epoch_loss))
            decliner.step()
            model.eval()
            with torch.no_grad():
                for x, y, c in dataload_test:
                    masks = y.to(int)
                    masks = masks.squeeze()
                    masks = torch.flatten(masks, start_dim=0, end_dim=1)
                    x = x.to(device)
                    outputs_test = model(x)
                    output_test = outputs_test[0].to('cpu')
                    output_test_f1 = output_test.squeeze()
                    output_test_f1 = torch.flatten(output_test_f1, start_dim=0, end_dim=1)
                    f1 = metric_f1(output_test_f1, masks)
                    hd95 = metric_hd95(output_test, c)
            f1 = metric_f1.compute()
            hd95 = metric_hd95.compute()
            if f1 > max_f1:
                max_f1 = f1
                model_state_dict = model.state_dict()
                torch.save(model_state_dict, save_path)
            print('f1:%0.4f' % f1)
            print('hd95:%0.4f' % hd95)
            metric_f1.reset()
            metric_hd95.reset()
        print('max_f1:%0.4f' % max_f1)

    elif args.model == "DecoupleUNet":
        criterion_relax = RelaxLoss(classes=2)
        criterion_focal_1 = FocalLoss(gamma=2, alpha=0.9, adaptive_weight=True)
        criterion_focal_2 = FocalLoss(gamma=2, alpha=0.9, adaptive_weight=True)
        criterion_dice = DiceLoss()
        for epoch in range(num_epochs):
            model.train()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            dataset_size = len(dataload.dataset)
            epoch_loss = 0
            step = 0
            for x, y, c in dataload:
                optimizer.zero_grad()
                inputs = x.to(device)
                masks = y.to(device)
                contours = c.to(device)
                outputs = model(inputs)

                loss1 = criterion_focal_1(outputs[0], masks)
                loss2 = criterion_focal_2(outputs[2], contours)
                loss3 = criterion_dice(outputs[2], contours)
                loss4 = criterion_relax(outputs[1], masks)

                loss = loss1 + loss2 + 0.01 * loss3 + loss4

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                step += 1
                print("%d/%d,train_loss:%0.4f" % (step, dataset_size // dataload.batch_size, loss.item()))
            print("epoch %d loss:%0.4f" % (epoch, epoch_loss))
            decliner.step()
            model.eval()
            with torch.no_grad():
                for x, y, c in dataload_test:
                    masks = y.to(int)
                    masks = masks.squeeze()
                    masks = torch.flatten(masks, start_dim=0, end_dim=1)
                    x = x.to(device)
                    outputs_test = model(x)
                    output_test = outputs_test[0].to('cpu')
                    output_test_f1 = output_test.squeeze()
                    output_test_f1 = torch.flatten(output_test_f1, start_dim=0, end_dim=1)
                    f1 = metric_f1(output_test_f1, masks)
                    hd95 = metric_hd95(output_test, c)
            f1 = metric_f1.compute()
            hd95 = metric_hd95.compute()
            if f1 > max_f1:
                max_f1 = f1
                model_state_dict = model.state_dict()
                torch.save(model_state_dict, save_path)
            print('f1:%0.4f' % f1)
            print('hd95:%0.4f' % hd95)
            metric_f1.reset()
            metric_hd95.reset()
        print('max_f1:%0.4f' % max_f1)
    return model


def train():
    # set network type
    if args.model == "U-Net":
        model = VanillaUNet(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "U-Net++":
        model = UNet_2Plus(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "Attention U-Net":
        model = AttentionUNet(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "GSCNN":
        model = GSCNN(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "DeeplabV3+":
        model = DeeplabV3Plus(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "DecoupleSegNet":
        model = DecoupleSegNet(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "DecoupleUNet":
        model = DecoupleUNet(in_ch=args.in_ch, out_ch=args.out_ch).to(device)

    # set batch size
    batch_size = args.batch_size
    # set optimizer(we used Adam)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # initialize the weight of network
    model.init_weight()
    # set dataloader for training
    image_dataset = ImageDataset("G:/python project/final/dataset/cell_patches", is_train=True, transform=x_transform,
                                 target_transform=y_transform)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # set dataloader for testing
    image_dataset_test = ImageDatasetNoTile("G:/python project/final/dataset/cell_original", transform=x_transform,
                                            target_transform=y_transform,
                                            is_train=False)
    dataloader_test = DataLoader(image_dataset_test, num_workers=2, batch_size=1)
    train_model(model, optimizer, dataloader, dataloader_test, num_epochs=args.epoch)


def test():
    if args.model == "U-Net":
        model = VanillaUNet(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "U-Net++":
        model = UNet_2Plus(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "Attention U-Net":
        model = AttentionUNet(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "GSCNN":
        model = GSCNN(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "DeeplabV3+":
        model = DeeplabV3Plus(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "DecoupleSegNet":
        model = DecoupleSegNet(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    elif args.model == "DecoupleUNet":
        model = DecoupleUNet(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    metric_f1 = F1(num_classes=1)
    metric_hd95 = HD95()
    model.load_state_dict(
        torch.load(args.weight, map_location='cuda:0'))
    image_dataset_test = ImageDatasetNoTile("G:/python project/final/dataset/cell_original", transform=x_transform,
                                            target_transform=y_transform,
                                            is_train=False)
    dataload_test = DataLoader(image_dataset_test, num_workers=2, batch_size=1)
    model.eval()
    loop = 0
    with torch.no_grad():
        for x, y, c in dataload_test:
            x = x.to(device)
            masks = y.to(int)
            masks = masks.squeeze()
            masks = torch.flatten(masks, start_dim=0, end_dim=1)
            outputs_test = model(x)

            if args.model == "U-Net" or args.model == "U-Net++" or args.model == "Attention U-Net" or args.model == "DeeplabV3+":
                output_test = outputs_test.to('cpu')
                img = torch.squeeze(outputs_test).to("cpu").numpy() * 255
            elif args.model == "GSCNN" or args.model == "DecoupleSegNet" or args.model == "DecoupleUNet":
                img = torch.squeeze(outputs_test[0]).to("cpu").numpy() * 255
                output_test = outputs_test[0].to('cpu')
            output_test_f1 = output_test.squeeze()
            output_test_f1 = torch.flatten(output_test_f1, start_dim=0, end_dim=1)
            f1 = metric_f1(output_test_f1, masks)
            hd95 = metric_hd95(output_test, c)

            loop += 1
            cv2.imwrite('G:/python project/final/mask_%d_1.png' % loop, img.astype('uint8'))

        f1 = metric_f1.compute()
        hd95 = metric_hd95.compute()
        print('f1:%0.4f' % f1)
        print('hd95:%0.4f' % hd95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train or test')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=int, default=0.0001)
    parser.add_argument('--weight', type=str, help='the path of the mode weight file')
    parser.add_argument('--model', type=str, default="DecoupleUNet", help='choose a model')
    parser.add_argument('--in_ch', type=int, default=3, help='the channel number of input')
    parser.add_argument('--out_ch', type=int, default=1, help='the channel number of output')
    parser.add_argument('--epoch', type=int, default=180, help='the epoch for training')
    args = parser.parse_args()

    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()

from tqdm import tqdm
import math

# train for one epoch
def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(train_optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for (images, _) in train_bar:
        im_1, im_2 = images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)
        
        train_optimizer.zero_grad()
        loss = net.forward(im_1, im_2, args)
        loss.backward()
        train_optimizer.step()
        
        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size

        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
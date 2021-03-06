import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import shutil
import skimage
from skimage import transform
import torch
import os
from collections import defaultdict, OrderedDict
# from html4vision import Col, imagetable
# from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import multiple_res_model

import util
import layers

input_size = 128
output_size = 192
expand_size = (output_size - input_size) // 2
patch_w = output_size // 8
patch_h = output_size // 8
patch = (1, patch_h, patch_w)


# class Generator(nn.Module):
#     def __init__(self, channels=3, extra_upsample=False):
#         super(Generator, self).__init__()
#         def downsample(in_feat, out_feat, normalize=True):
#             layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
#             if normalize:
#                 layers.append(nn.BatchNorm2d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2))
#             return layers

#         def upsample(in_feat, out_feat, normalize=True):
#             layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
#             if normalize:
#                 layers.append(nn.BatchNorm2d(out_feat, 0.8))
#             layers.append(nn.ReLU())
#             return layers

#         self.down1 = downsample(channels, 64, normalize=True)

#         self.down2 = downsample(64, 64, normalize=True)

#         self.down3 = downsample(64, 128, normalize=True)

#         self.down4 = downsample(128, 256, normalize=True)

#         self.down5 = downsample(256, 512, normalize=True)

#         self.down6 = downsample(512, 4000, normalize=True)

#         self.down_middle = nn.Conv2d(512, 4000, 1)

#         self.up1 = upsample(4000, 512)
#         self.up2 = upsample(512, 256)
#         self.up3 = upsample(256, 128)
#         self.up4 = upsample(128, 64)


#         self.final_down = nn.Conv2d(64, channels, 3, 1, 1)
#         self.final_activation = nn.Tanh()

#     def forward(self, x):
#         z = self.down1[0](x)
#         z = self.down1[1](z)
#         residual = z
#         for i in range(3):
#             z = self.down2[i](z)
#         for i in range(3):
#             z = self.down3[i](z)
#         for i in range(3):
#             z = self.down4[i](z)
#         for i in range(3):
#             z = self.down5[i](z)
#         for i in range(3):
#             z = self.down6[i](z)
#         z += residual
#         z = self.down_middle(z)

#         for i in range(3):
#             z = self.up1[i](z)
#         for i in range(3):
#             z = self.up2[i](z)
#         for i in range(3):
#             z = self.up3[i](z)
#         for i in range(3):
#             z = self.up4[i](z)

#         z = self.final_down(z)
#         z = self.final_activation(z)
#         return z

class GlobalDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super(GlobalDiscriminator, self).__init__()
        self.output_shape = (24, 24)

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class LocalDiscriminator(nn.Module):
    def __init__(self, mask, channels=3):
        super(LocalDiscriminator, self).__init__()
        self.output_shape = (24, 24)

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img, mask):
        img = img * mask
        return self.model(img)


class ContextDiscriminator(nn.Module):
    def __init__(self, local_input_shape, global_input_shape, arc='places2'):
        super(ContextDiscriminator, self).__init__()
        self.arc = arc
        self.input_shape = [local_input_shape, global_input_shape]
        # TODO: For outpainting, local and global discriminator shapes are equal
        assert(local_input_shape == global_input_shape)
        self.output_shape = (1,)

        # self.model_ld = LocalDiscriminator(local_input_shape)
        self.model_ld = LocalDiscriminator(local_input_shape)

        # self.model_gd = GlobalDiscriminator(global_input_shape, arc=arc)
        self.model_gd = GlobalDiscriminator()

        # TODO: Remove, this stuff gets handled afterwards
        self.concat1 = layers.Concatenate(dim=-1)
        self.flatten1 = nn.Flatten()
        in_features = self.model_ld.output_shape[-1] ** 2 + self.model_gd.output_shape[-1] ** 2
        self.linear1 = nn.Linear(in_features, 1)
        self.act1 = nn.Sigmoid()

    def forward(self, x, mask):
        x_ld = self.model_ld(x, mask)
        x_gd = self.model_gd(x)
        # concat = self.concat1([self.flatten1(x_ld), self.flatten1(x_gd)])
        # print('concatenated and flattened discriminator outputs', concat.shape)
        # lin = self.linear1(concat)
        # out = self.act1(lin)
        out = (x_ld + x_gd) / 2
        return out


def construct_masked(input_img):
    resized = skimage.transform.resize(input_img, (input_size, input_size), anti_aliasing=True)
    result = np.ones((output_size, output_size))
    result[expand_size:-expand_size, expand_size:-expand_size, :] = resized
    return result


def blend_result(output_img, input_img, blend_width=8):
    '''
    Blends an input of arbitrary resolution with its output, using the highest resolution of both.
    Returns: final result + source mask.
    '''
    print('Input size:', input_img.shape)
    print('Output size:', output_img.shape)
    in_factor = input_size / output_size
    if input_img.shape[1] < in_factor * output_img.shape[1]:
        # Output dominates, adapt input
        out_width, out_height = output_img.shape[1], output_img.shape[0]
        in_width, in_height = int(out_width * in_factor), int(out_height * in_factor)
        input_img = skimage.transform.resize(input_img, (in_height, in_width), anti_aliasing=True)
    else:
        # Input dominates, adapt output
        in_width, in_height = input_img.shape[1], input_img.shape[0]
        out_width, out_height = int(in_width / in_factor), int(in_height / in_factor)
        output_img = skimage.transform.resize(output_img, (out_height, out_width), anti_aliasing=True)

    # Construct source mask
    src_mask = np.zeros((output_size, output_size))
    src_mask[expand_size+1:-expand_size-1, expand_size+1:-expand_size-1] = 1 # 1 extra pixel for safety
    src_mask = distance_transform_edt(src_mask) / blend_width
    src_mask = np.minimum(src_mask, 1)
    src_mask = skimage.transform.resize(src_mask, (out_height, out_width), anti_aliasing=True)
    src_mask = np.tile(src_mask[:, :, np.newaxis], (1, 1, 3))

    # Pad input
    input_pad = np.zeros((out_height, out_width, 3))
    x1 = (out_width - in_width) // 2
    y1 = (out_height - in_height) // 2
    input_pad[y1:y1+in_height, x1:x1+in_width, :] = input_img

    # Merge
    blended = input_pad * src_mask + output_img * (1 - src_mask)

    print('Blended size:', blended.shape)

    return blended, src_mask


def perform_outpaint(gen_model, input_img, blend_width=8):
    '''
    Performs outpainting on a single color image with arbitrary dimensions.
    Returns: 192x192 unmodified output + upscaled & blended output.
    '''
    # Enable evaluation mode
    gen_model.eval()
    torch.set_grad_enabled(False)

    # Construct masked input
    resized = skimage.transform.resize(input_img, (input_size, input_size), anti_aliasing=True)
    masked_img = np.ones((output_size, output_size, 3))
    new_channel=np.ones((192,192,4))
    new_channel[:,:,:3] = masked_img
    masked_img=new_channel
    print(masked_img.shape)
    masked_img[expand_size:-expand_size, expand_size:-expand_size, :3] = resized
    assert(masked_img.shape[0] == output_size)
    assert(masked_img.shape[1] == output_size)
    assert(masked_img.shape[2] == 4)

    # Convert to torch
    masked_img = masked_img.transpose(2, 0, 1)
    # print(masked_img.shape)
    masked_img = torch.tensor(masked_img[np.newaxis], dtype=torch.float)
    print(masked_img.size())

    # Call generator
    output_img = gen_model(masked_img)

    # Convert to numpy
    output_img = output_img.cpu().numpy()
    output_img = output_img.squeeze().transpose(1, 2, 0)
    output_img = np.clip(output_img, 0, 1)

    # Blend images
    norm_input_img = input_img.copy().astype('float')
    if np.max(norm_input_img) > 1:
        norm_input_img /= 255
    blended_img, src_mask = blend_result(output_img, norm_input_img)
    blended_img = np.clip(blended_img, 0, 1)

    return output_img, blended_img


def load_model(model_path):
    model = multiple_res_model.CompletionNetwork()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Remove 'module' if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:] # remove 'module'
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.cpu()
    model.eval()
    return model


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class CEImageDataset(Dataset):

    def __init__(self, root, transform, output_size=192, input_size=128, outpaint=True):
        self.transform = transform
        self.output_size = output_size
        self.input_size = input_size
        self.outpaint = outpaint
        self.files = sorted(glob.glob("%s/*.jpg" % root))

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.output_size - self.input_size) // 2

        if not self.outpaint:
            # masked_part = img[:, i:i + self.input_size, i:i + self.input_size]
            masked_part = img[:, i:i + self.input_size,:]
            masked_img = img.clone()
            masked_img[:, i:i + self.input_size,:] = 1
        else:
            masked_part = -1  # ignore this for outpainting
            masked_img = img.clone()
            masked_img[:, :i, :] = 1
            masked_img[:, -i:, :] = 1
            masked_img[:, :, :i] = 1
            masked_img[:, :, -i:] = 1

        return masked_img, masked_part

    def __getitem__(self, index):

        try:
            img = Image.open(self.files[index % len(self.files)]).convert('RGB')
            img = self.transform(img)
        except:
            # Likely corrupt image file, so generate black instead
            img = torch.zeros((3, self.output_size, self.output_size))

        masked_img, masked_part = self.apply_center_mask(img)

        return img, masked_img, masked_part

    def __len__(self):
        return len(self.files)


def is_power_two(n):
    mod = np.mod(np.log(n) / np.log(2), 1)
    return mod < 1e-9 or mod > 1 - 1e-9


def finish_inpaint(imgs, outputs):
    global output_size, input_size
    result = imgs.clone()
    x1 = (output_size - input_size) // 2
    x2 = x1 + input_size
    y1 = (output_size - input_size) // 2
    y2 = y1 + input_size
    result[:, :, y1:y2, x1:x2] = outputs
    return result


def generate_html(G_net, D_net, mask, device, data_loaders, html_save_path, max_rows=64):
    '''
    Visualizes one batch from both the training and validation sets.
    Images are stored in the specified HTML file path.
    '''
    G_net.eval()
    D_net.eval()
    torch.set_grad_enabled(False)
    if os.path.exists(html_save_path):
        shutil.rmtree(html_save_path)
    os.makedirs(html_save_path + '/images')

    # Evaluate examples
    for phase in ['train', 'val']:
        imgs, masked_imgs, masked_parts = next(iter(data_loaders[phase]))

        masked_imgs = masked_imgs.to(device)
        mask_shape = (masked_imgs.shape[0], 1, masked_imgs.shape[2], masked_imgs.shape[3])
        mask = util.gen_mask(mask_shape).to(device)

        masked_imgs = torch.cat((masked_imgs, mask), dim=1).to(device)
        outputs = G_net(masked_imgs)
        masked_imgs = masked_imgs.cpu()
        # if not outpaint:
        #     results = finish_inpaint(imgs, outputs.cpu())
        # else:
        results = outputs.cpu()
        # Store images
        for i in range(min(imgs.shape[0], max_rows)):
            save_image(masked_imgs[i][:3], html_save_path + '/images/' + phase + '_' + str(i) + '_masked.jpg')
            save_image(results[i], html_save_path + '/images/' + phase + '_' + str(i) + '_result.jpg')
            save_image(imgs[i], html_save_path + '/images/' + phase + '_' + str(i) + '_truth.jpg')

    # Generate table
    cols = [
        Col('id1', 'ID'),
        Col('img', 'Training set masked', html_save_path + '/images/train_*_masked.jpg'),
        Col('img', 'Training set result', html_save_path + '/images/train_*_result.jpg'),
        Col('img', 'Training set truth', html_save_path + '/images/train_*_truth.jpg'),
        Col('img', 'Validation set masked', html_save_path + '/images/val_*_masked.jpg'),
        Col('img', 'Validation set result', html_save_path + '/images/val_*_result.jpg'),
        Col('img', 'Validation set truth', html_save_path + '/images/val_*_truth.jpg'),
    ]
    imagetable(cols, out_file=html_save_path + '/index.html',
               pathrep=(html_save_path + '/images', 'images'))
    print('Generated image table at: ' + html_save_path + '/index.html')


def get_adv_weight(adv_weight, epoch):
    if isinstance(adv_weight, list):
        if epoch < 10:
            return adv_weight[0]
        elif epoch < 30:
            return adv_weight[1]
        elif epoch < 60:
            return adv_weight[2]
        else:
            return adv_weight[3]
    else:  # just one number
        return adv_weight


def train(G_net, D_net, device, criterion_pxl, criterion_D, optimizer_G, optimizer_D, data_loaders,
          model_save_path, html_save_path, n_epochs=200, start_epoch=0, adv_weight=0.001):
    """
    Outpainting GAN training loop based on Context Encoder implementation in PyTorch.

    :param G_net:
    :param D_net:
    :param device:
    :param criterion_pxl:
    :param criterion_D:
    :param optimizer_G:
    :param optimizer_D:
    :param data_loaders:
    :param model_save_path:
    :param html_save_path:
    :param n_epochs:
    :param start_epoch:
    :param adv_weight:
    :return:
    """
    Tensor = torch.cuda.FloatTensor
    hist_loss = defaultdict(list)

    for epoch in range(start_epoch, n_epochs):

        for phase in ['train', 'val']:
            batches_done = 0

            running_loss_pxl = 0.0
            running_loss_adv = 0.0
            running_loss_D = 0.0

            for idx, (imgs, masked_imgs, masked_parts) in enumerate(data_loaders[phase]):
                if phase == 'train':
                    G_net.train()
                    D_net.train()
                else:
                    G_net.eval()
                    D_net.eval()
                torch.set_grad_enabled(phase == 'train')

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False).to(device)
                fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False).to(device)
                # Configure input
                imgs = Variable(imgs.type(Tensor)).to(device)
                masked_imgs = Variable(masked_imgs.type(Tensor)).to(device)
                # Concatenate mask as 4th channel

                mask_shape = (masked_imgs.shape[0], 1, masked_imgs.shape[2], masked_imgs.shape[3])
                mask = util.gen_mask(mask_shape).to(device)
                masked_imgs = torch.cat((masked_imgs, mask), dim=1)
                # print("Masked images size:",masked_imgs.size())

                # if not(outpaint):
                #     masked_parts = Variable(masked_parts.type(Tensor)).to(device)

                # -----------
                #  Generator
                # -----------
                if phase == 'train':
                    optimizer_G.zero_grad()
                # Generate a batch of images
                outputs = G_net(masked_imgs)
                # Adversarial and pixelwise loss
                # if not(outpaint):
                #     loss_pxl = criterion_pxl(outputs, masked_parts)  # inpaint: compare center part only
                # else:
                # print("Generated output images size:",outputs.size())
                # plt.imshow( outputs[0].permute(1, 2, 0)  )
                # print("Ground truth imgs size:",imgs.size())
                # plt.imshow( imgs[0].permute(1, 2, 0)  )
                loss_pxl = criterion_pxl(outputs, imgs)  # outpaint: compare to full ground truth
                loss_adv = criterion_D(D_net(outputs, mask), valid)
                # Total loss
                cur_adv_weight = get_adv_weight(adv_weight, epoch)
                loss_G = (1 - cur_adv_weight) * loss_pxl + cur_adv_weight * loss_adv
                if phase == 'train':
                    loss_G.backward()
                    optimizer_G.step()

                # ---------------
                #  Discriminator
                # ---------------
                if phase == 'train':
                    optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                # if not(outpaint):
                #     real_loss = criterion_D(D_net(masked_parts), valid) # inpaint: check center part only
                # else:
                # TODO: need to clarify real vs fake loss here
                real_loss = criterion_D(D_net(imgs, mask), valid)  # outpaint: check full ground truth
                fake_loss = criterion_D(D_net(outputs.detach(), mask), fake)
                loss_D = 0.5 * (real_loss + fake_loss)
                if phase == 'train':
                    loss_D.backward()
                    optimizer_D.step()

                # Update & print statistics
                batches_done += 1
                running_loss_pxl += loss_pxl.item()
                running_loss_adv += loss_adv.item()
                running_loss_D += loss_D.item()
                if phase == 'train' and is_power_two(batches_done):
                    print('Batch {:d}/{:d}  loss_pxl {:.4f}  loss_adv {:.4f}  loss_D {:.4f}'.format(
                          batches_done, len(data_loaders[phase]), loss_pxl.item(), loss_adv.item(), loss_D.item()))

            # Store model & visualize examples
            if phase == 'train':
                # if not os.path.exists(model_save_path):
                #     os.makedirs(model_save_path)
                # torch.save(G_net.state_dict(), model_save_path + '/G_' + str(epoch) + '.pt')
                # torch.save(D_net.state_dict(), model_save_path + '/D_' + str(epoch) + '.pt')
                generate_html(G_net, D_net, mask, device, data_loaders, html_save_path + '/' + str(epoch))

            # Store & print statistics
            cur_loss_pxl = running_loss_pxl / batches_done
            cur_loss_adv = running_loss_adv / batches_done
            cur_loss_D = running_loss_D / batches_done
            hist_loss[phase + '_pxl'].append(cur_loss_pxl)
            hist_loss[phase + '_adv'].append(cur_loss_adv)
            hist_loss[phase + '_D'].append(cur_loss_D)
            print('Epoch {:d}/{:d}  {:s}  loss_pxl {:.4f}  loss_adv {:.4f}  loss_D {:.4f}'.format(
                  epoch + 1, n_epochs, phase, cur_loss_pxl, cur_loss_adv, cur_loss_D))

        print()

    print('Done!')
    return hist_loss

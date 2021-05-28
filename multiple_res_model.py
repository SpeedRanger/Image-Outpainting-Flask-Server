import torch
import torch.nn as nn
import torch.nn.functional as F

import layers


class IdentityExpansion(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityExpansion, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out

class Dense_Block(nn.Module):
    def __init__(self, in_channels):
        super(Dense_Block, self).__init__()
        # self.layer_256=layer_256
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_features = in_channels)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        # if self.layer_256==True:
        #     self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        bn = self.bn1(x) 
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        # Concatenate in channel dimension
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(c3_dense)) 
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))
        
        # if self.layer_256==True:
        #     conv6 = self.relu(self.conv6(c5_dense))
        #     c6_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6], 1))
        
        # if self.layer_256==True:
        #     return c6_dense
        # else:
        #     return c5_dense
        return c5_dense
        
    
class Dense_Block_new(nn.Module):
    def __init__(self, in_channels):
        super(Dense_Block_new, self).__init__()
        # self.layer_256=layer_256
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_features = in_channels)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 160, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv7 = nn.Conv2d(in_channels = 192, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 224, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        # self.conv9 = nn.Conv2d(in_channels = 256, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        bn = self.bn1(x) 
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        # Concatenate in channel dimension
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(c3_dense)) 
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))
        
        conv6 = self.relu(self.conv6(c5_dense))
        c6_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6], 1))
        
        conv7 = self.relu(self.conv7(c6_dense))
        c7_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7], 1))
        
        conv8 = self.relu(self.conv8(c7_dense))
        c8_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8], 1))
        
        # conv9 = self.relu(self.conv9(c8_dense))
        # c9_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9], 1))
        
        return c8_dense

class Transition_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer, self).__init__() 

        self.relu = nn.ReLU(inplace = True) 
        self.bn = nn.BatchNorm2d(num_features = out_channels) 
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False) 
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)   
        
    def forward(self, x): 
        bn = self.bn(self.relu(self.conv(x))) 
        out = self.avg_pool(bn) 
        return out 



class CompletionNetwork(nn.Module):
    def __init__(self):
        super(CompletionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.ReLU()
        self.denseblock1 = self._make_dense_block(Dense_Block, 64) 
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128)
        # self.denseblock2 = self._make_dense_block(Dense_Block, 128)
        # self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128)
        # self.denseblock3 = self._make_dense_block(Dense_Block_new, 128)
        # self.transitionLayer3 = self._make_transition_layer(Transition_Layer, in_channels = 256, out_channels = 128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.act7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)
        self.act8 = nn.ReLU()
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)
        self.act9 = nn.ReLU()
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=16, padding=16)
        self.act10 = nn.ReLU()
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act11 = nn.ReLU()
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act12 = nn.ReLU()
        # Here lies the latent space
        self.deconv13 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.act13 = nn.ReLU()
        # # The channel-wise fully-connected layer is followed by a 1 stride convolution layer to propagate information across channels
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.act14 = nn.ReLU()
        self.deconv15 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.act15 = nn.ReLU()
        self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.act16 = nn.ReLU()
        self.conv17 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.act17 = nn.Sigmoid()

    def _make_dense_block(self, block, in_channels): 
        layers = [] 
        layers.append(block(in_channels)) 
        return nn.Sequential(*layers)
        
    def _make_transition_layer(self, layer, in_channels, out_channels): 
        modules = [] 
        modules.append(layer(in_channels, out_channels)) 
        return nn.Sequential(*modules) 

    def forward(self, x):
        out = self.act1(self.conv1(x))     
        out = self.denseblock1(out)
        # print("Output after dense 1:",out.size())
        out = self.transitionLayer1(out)
        # print("Output after transition 1:",out.size())
        # out = self.denseblock2(out)
        # print("Output after dense:",out.size())
        # out = self.transitionLayer2(out)
        # print("Output after transition:",out.size())
        # out = self.denseblock3(out)
        # print("Output after dense:",out.size())
        # out = self.transitionLayer3(out)
        # print("Output after transition:",out.size())
        out = self.act4(self.conv4(out))
        res3 = out
        out = self.act5(self.conv5(out))
        out = self.act6(self.conv6(out))
        out+=res3

        res4 = out
        out = self.act7(self.conv7(out))
        out = self.act8(self.conv8(out))
        out+=res4

        res5 = out
        out = self.act9(self.conv9(out))
        out = self.act10(self.conv10(out))
        out+=res5
        out = self.act11(self.conv11(out))
        # print("Output after conv:",out.size())
        out = self.act12(self.conv12(out))
        # print("Output after conv:",out.size())
        out = self.act13(self.deconv13(out))
        # print("Output after deconv:",out.size())
        out = self.act14(self.conv14(out))
        # print("Output after conv:",out.size())
        out = self.act15(self.deconv15(out))
        # print("Output after deconv:",out.size())
        out = self.act16(self.conv16(out))
        # print("Output after conv:",out.size())
        out = self.act17(self.conv17(out))
        # print("Output after conv:",out.size())
        # print("Outout final:",out.size())
        return out



class LocalDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]
        self.conv1 = nn.Conv2d(self.img_c, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()
        in_features = 512 * (self.img_h//32) * (self.img_w//32)
        self.flatten6 = layers.Flatten()
        self.linear6 = nn.Linear(in_features, 1024)
        self.act6 = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.act6(self.linear6(self.flatten6(x)))
        return x


class GlobalDiscriminator(nn.Module):
    def __init__(self, input_shape, arc='places2'):
        super(GlobalDiscriminator, self).__init__()
        self.arc = arc
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]

        self.conv1 = nn.Conv2d(self.img_c, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()
        if arc == 'celeba':
            in_features = 512 * (self.img_h//32) * (self.img_w//32)
            self.flatten6 = layers.Flatten()
            self.linear6 = nn.Linear(in_features, 1024)
            self.act6 = nn.ReLU()
        elif arc == 'places2':
            self.conv6 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
            self.bn6 = nn.BatchNorm2d(512)
            self.act6 = nn.ReLU()
            in_features = 512 * (self.img_h//64) * (self.img_w//64)
            self.flatten7 = layers.Flatten()
            self.linear7 = nn.Linear(in_features, 1024)
            self.act7 = nn.ReLU()
        else:
            raise ValueError('Unsupported architecture \'%s\'.' % self.arc)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        if self.arc == 'celeba':
            x = self.act6(self.linear6(self.flatten6(x)))
        elif self.arc == 'places2':
            x = self.bn6(self.act6(self.conv6(x)))
            x = self.act7(self.linear7(self.flatten7(x)))
        return x

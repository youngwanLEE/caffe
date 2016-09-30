import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in xrange(0, num):
        ret.append(var[0])
    else:
      for i in xrange(0, num):
        ret.append(var)
    return ret

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, eps=0.001,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias'):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, **kwargs)
  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def ResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1, dilation=1):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  branch_name = 'branch2a'
  ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  if dilation == 1:
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  else:
    pad = int((3 + (dilation - 1) * 2) - 1) / 2
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=pad, stride=1, use_scale=use_scale,
        dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2c'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
      num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)


def InceptionTower(net, from_layer, tower_name, layer_params):
  use_scale = False
  for param in layer_params:
    tower_layer = '{}/{}'.format(tower_name, param['name'])
    del param['name']
    if 'pool' in tower_layer:
      net[tower_layer] = L.Pooling(net[from_layer], **param)
    else:
      ConvBNLayer(net, from_layer, tower_layer, use_bn=True, use_relu=True,
          use_scale=use_scale, **param)
    from_layer = tower_layer
  return net[from_layer]

def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, label_map_file='',
        transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    if output_label:
        data, label = L.AnnotatedData(name="data",
            annotated_data_param=dict(label_map_file=label_map_file,
                batch_sampler=batch_sampler),
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=2, **kwargs)
        return [data, label]
    else:
        data = L.AnnotatedData(name="data",
            annotated_data_param=dict(label_map_file=label_map_file,
                batch_sampler=batch_sampler),
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=1, **kwargs)
        return data


def VGGNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool4'
        net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv5_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=6, kernel_size=7, dilation=2, **kwargs)
            else:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=3, kernel_size=3, dilation=3, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=3, kernel_size=7, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net


def ResNet101Body(net, from_layer, use_pool5=True, use_dilation_conv5=False):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True)

    from_layer = 'res3a'
    for i in xrange(1, 4):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True)

    from_layer = 'res4a'
    for i in xrange(1, 23):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
      stride = 1
      dilation = 2

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=stride, use_branch1=True, dilation=dilation)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net


def ResNet152Body(net, from_layer, use_pool5=True, use_dilation_conv5=False):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True)

    from_layer = 'res3a'
    for i in xrange(1, 8):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True)

    from_layer = 'res4a'
    for i in xrange(1, 36):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
      stride = 1
      dilation = 2

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=stride, use_branch1=True, dilation=dilation)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net


def InceptionV3Body(net, from_layer, output_pred=False):
  # scale is fixed to 1, thus we ignore it.
  use_scale = False

  out_layer = 'conv'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=2, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'conv_1'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'conv_2'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=64, kernel_size=3, pad=1, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'pool'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  out_layer = 'conv_3'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=80, kernel_size=1, pad=0, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'conv_4'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=192, kernel_size=3, pad=0, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'pool_1'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  # inceptions with 1x1, 3x3, 5x5 convolutions
  for inception_id in xrange(0, 3):
    if inception_id == 0:
      out_layer = 'mixed'
      tower_2_conv_num_output = 32
    else:
      out_layer = 'mixed_{}'.format(inception_id)
      tower_2_conv_num_output = 64
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=48, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=64, kernel_size=5, pad=2, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
        dict(name='conv_2', num_output=96, kernel_size=3, pad=1, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=tower_2_conv_num_output, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3(in sequence) convolutions
  out_layer = 'mixed_3'
  towers = []
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=384, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
      dict(name='conv_2', num_output=96, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  # inceptions with 1x1, 7x1, 1x7 convolutions
  for inception_id in xrange(4, 8):
    if inception_id == 4:
      num_output = 128
    elif inception_id == 5 or inception_id == 6:
      num_output = 160
    elif inception_id == 7:
      num_output = 192
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        ])
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_2', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_3', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_4', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        ])
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3, 1x7, 7x1 filters
  out_layer = 'mixed_8'
  towers = []
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=320, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}/tower_1'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
      dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
      dict(name='conv_3', num_output=192, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  for inception_id in xrange(9, 11):
    num_output = 384
    num_output2 = 448
    if inception_id == 9:
      pool = P.Pooling.AVE
    else:
      pool = P.Pooling.MAX
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=320, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)

    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        ])
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output2, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=3, pad=1, stride=1),
        ])
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=pool, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  if output_pred:
    net.pool_3 = L.Pooling(net[from_layer], pool=P.Pooling.AVE, kernel_size=8, pad=0, stride=1)
    net.softmax = L.InnerProduct(net.pool_3, num_output=1008)
    net.softmax_prob = L.Softmax(net.softmax)

  return net

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True,
        min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], share_location=True, flip=True, clip=True,
        inter_layer_depth=0, kernel_size=1, pad=0, conf_postfix='', loc_postfix=''):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth > 0:
            inter_name = "{}_inter".format(from_layer)
            ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True,
                num_output=inter_layer_depth, kernel_size=3, pad=1, stride=1)
            from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        if max_sizes and max_sizes[i]:
            num_priors_per_location = 2 + len(aspect_ratio)
        else:
            num_priors_per_location = 1 + len(aspect_ratio)
        if flip:
            num_priors_per_location += len(aspect_ratio)

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        if max_sizes and max_sizes[i]:
            if aspect_ratio:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i], max_size=max_sizes[i],
                    aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
            else:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i], max_size=max_sizes[i],
                    clip=clip, variance=prior_variance)
        else:
            if aspect_ratio:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i],
                    aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
            else:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i],
                    clip=clip, variance=prior_variance)
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers


#ResNet50Layer use bias in conv1 and freezed
def ConvBNLayerWithBias(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, use_scale=True, eps=0.001, conv_prefix='', conv_postfix='',
    bn_prefix='', bn_postfix='_bn', scale_prefix='', scale_postfix='_scale',
    bias_prefix='', bias_postfix='_bias'):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = { #freezed[LYW]
        'param': [dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)],
        'weight_filler': dict(type='gaussian', std=0.01),
        #'bias_term': True,
        'bias_filler': dict(type='constant', value=0),
        }
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, **kwargs)
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)    

# Freezed ResNet in Jun_16_2016 by youngwan
def ConvBNLayerFreeze(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, use_scale=True, eps=0.001, conv_prefix='', conv_postfix='',
    bn_prefix='', bn_postfix='_bn', scale_prefix='', scale_postfix='_scale',
    bias_prefix='', bias_postfix='_bias',Freeze=True):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=0, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, **kwargs)
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def CreateAnnotatedDataLayerLEVELDB(source, batch_size=100, backend=P.Data.LEVELDB,
        output_label=True, train=True,
        transform_param={}):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    if output_label:
        data, label = L.Data(name="data",
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=2, **kwargs)
        return [data, label]
    else:
        data = L.Data(name="data",
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=1, **kwargs)
        return data

# For ImageNet by LYW 2016.09.26.
def CreateAnnotatedDataLayerLMDB(source, batch_size=100, backend=P.Data.LMDB,
        output_label=True, train=True,
        transform_param={}):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    if output_label:
        data, label = L.Data(name="data",
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=2, **kwargs)
        return [data, label]
    else:
        data = L.Data(name="data",
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=1, **kwargs)
        return data



# Freezed ResNet in Jun_16_2016 by youngwan
def ResBodyFreeze(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1,Freeze=True):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayerFreeze(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  branch_name = 'branch2a'
  ConvBNLayerFreeze(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  ConvBNLayerFreeze(net, out_name, branch_name, use_bn=True, use_relu=True,
      num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2c'
  ConvBNLayerFreeze(net, out_name, branch_name, use_bn=True, use_relu=False,
      num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2]) #layer saved!
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)


def ResBasic333Body(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  branch_name = 'branch2a'
  ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
      num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2c'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
      num_output=out2c, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2]) #layer saved!
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)

def ResBasic33Body(net, from_layer, block_name, out2a, out2b, stride, use_branch1):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2b, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  branch_name = 'branch2a'
  ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
      num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2]) #layer saved!
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)

## for CIFAR-10 by LYW : ResNet 19 Layers with only bottleneck conv 3x3 in 12th_Sep_2016

def ResNet19_Conv3x3_Cifar10(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 300 --> 150
        num_output=64, kernel_size=3, pad=1, stride=1,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )

    #net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2) # 150 --> 75 : in July 5th 2016 by LYW

    from_layer = 'conv1_relu'
    ResBasic333Body(net, from_layer, '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True) # 75
    ResBasic333Body(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    ResBasic333Body(net, 'res2b', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True) # 75 --> 38
    ResBasic333Body(net, 'res3a', '3d', out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False) # 38

    from_layer = 'res3d'
    ResBasic333Body(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBasic333Body(net, 'res4a', '4b', out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4b_relu, pool=P.Pooling.AVE, global_pooling=True)

    net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net

## for CIFAR-10 by LYW : ResNet 19 Layers with only basic conv 3x3 in 12th_Sep_2016

def ResNet19_Conv3x3_basic_l4_Cifar10(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 300 --> 150
        num_output=64, kernel_size=3, pad=1, stride=1,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )

    #net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2) # 150 --> 75 : in July 5th 2016 by LYW

    from_layer = 'conv1_relu'
    ResBasic33Body(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=True) # 75
    ResBasic33Body(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResBasic33Body(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    ResBasic33Body(net, 'res2c', '3a', out2a=512, out2b=512,  stride=2, use_branch1=True) # 75 --> 38
    ResBasic33Body(net, 'res3a', '3b', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38
    ResBasic33Body(net, 'res3b', '3c', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38

    from_layer = 'res3c'
    ResBasic33Body(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBasic33Body(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)
    ResBasic33Body(net, 'res4b', '4c', out2a=256, out2b=256,  stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4c_relu, pool=P.Pooling.AVE, global_pooling=True)

    net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net

def ResNet_15_Conv3x3_basic_l4_Cifar10(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 300 --> 150
        num_output=64, kernel_size=3, pad=1, stride=1,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )

    #net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2) # 150 --> 75 : in July 5th 2016 by LYW

    from_layer = 'conv1_relu'
    ResBasic33Body(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=True) # 75
    ResBasic33Body(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResBasic33Body(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    ResBasic33Body(net, 'res2c', '3a', out2a=512, out2b=512,  stride=2, use_branch1=True) # 75 --> 38
    ResBasic33Body(net, 'res3a', '3b', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38
    #ResBasic33Body(net, 'res3b', '3c', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38

    from_layer = 'res3b'
    ResBasic33Body(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBasic33Body(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)
    #ResBasic33Body(net, 'res4b', '4c', out2a=256, out2b=256,  stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4b_relu, pool=P.Pooling.AVE, global_pooling=True)

    net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net

# For Cifar-100 by LYW in 2016.09.25.
def ResNet_15_Conv3x3_basic_l4_Cifar100(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 300 --> 150
        num_output=64, kernel_size=3, pad=1, stride=1,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )

    #net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2) # 150 --> 75 : in July 5th 2016 by LYW

    from_layer = 'conv1_relu'
    ResBasic33Body(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=True) # 75
    ResBasic33Body(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResBasic33Body(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    ResBasic33Body(net, 'res2c', '3a', out2a=512, out2b=512,  stride=2, use_branch1=True) # 75 --> 38
    ResBasic33Body(net, 'res3a', '3b', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38
    #ResBasic33Body(net, 'res3b', '3c', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38

    from_layer = 'res3b'
    ResBasic33Body(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBasic33Body(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)
    #ResBasic33Body(net, 'res4b', '4c', out2a=256, out2b=256,  stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4b_relu, pool=P.Pooling.AVE, global_pooling=True)

    net.fc100 = L.InnerProduct(net.pool5, num_output=100, weight_filler=dict(type='xavier'))

    return net

# For ImageNet by LYW in 2016.09.26.
def ResNet_15_Conv3x3_basic_l4_ImageNet(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 300 --> 150
        num_output=64, kernel_size=3, pad=1, stride=1,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )

    #net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2) # 150 --> 75 : in July 5th 2016 by LYW

    from_layer = 'conv1_relu'
    ResBasic33Body(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=True) # 75
    ResBasic33Body(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResBasic33Body(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    ResBasic33Body(net, 'res2c', '3a', out2a=512, out2b=512,  stride=2, use_branch1=True) # 75 --> 38
    ResBasic33Body(net, 'res3a', '3b', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38
    #ResBasic33Body(net, 'res3b', '3c', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38

    from_layer = 'res3b'
    ResBasic33Body(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBasic33Body(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)
    #ResBasic33Body(net, 'res4b', '4c', out2a=256, out2b=256,  stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4b_relu, pool=P.Pooling.AVE, global_pooling=True)

    net.fc1000 = L.InnerProduct(net.pool5, num_output=1000, weight_filler=dict(type='xavier'))

    return net


# for SSD by LYW in 2016. 09. 24.
def ResNet_15_Conv3x3_basic_l3_SSD(net, from_layer, global_pool=False):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 300 --> 150
        num_output=64, kernel_size=3, pad=1, stride=2,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )

    #net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2) # 150 --> 75 : in July 5th 2016 by LYW

    from_layer = 'conv1_relu'
    ResBasic33Body(net, from_layer, '2a', out2a=64, out2b=64, stride=2, use_branch1=True) # 75
    ResBasic33Body(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResBasic33Body(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    ResBasic33Body(net, 'res2c', '3a', out2a=384, out2b=384,  stride=2, use_branch1=True) # 75 --> 38
    ResBasic33Body(net, 'res3a', '3b', out2a=384, out2b=384,  stride=1, use_branch1=False) # 38
    #ResBasic33Body(net, 'res3b', '3c', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38

    from_layer = 'res3b'
    ResBasic33Body(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBasic33Body(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)
    #ResBasic33Body(net, 'res4b', '4c', out2a=256, out2b=256,  stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4b_relu, pool=P.Pooling.AVE, global_pooling=True)
      net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net

# for SSD by LYW in 2016. 09. 24.
def ResNet_15_Conv3x3_basic_l4_SSD(net, from_layer, global_pool=False):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 300 --> 150
        num_output=64, kernel_size=3, pad=1, stride=2,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )

    #net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2) # 150 --> 75 : in July 5th 2016 by LYW

    from_layer = 'conv1_relu'
    ResBasic33Body(net, from_layer, '2a', out2a=64, out2b=64, stride=2, use_branch1=True) # 75
    ResBasic33Body(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResBasic33Body(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    ResBasic33Body(net, 'res2c', '3a', out2a=512, out2b=512,  stride=2, use_branch1=True) # 75 --> 38
    ResBasic33Body(net, 'res3a', '3b', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38
    #ResBasic33Body(net, 'res3b', '3c', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38

    from_layer = 'res3b'
    ResBasic33Body(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBasic33Body(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)
    #ResBasic33Body(net, 'res4b', '4c', out2a=256, out2b=256,  stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4b_relu, pool=P.Pooling.AVE, global_pooling=True)
      net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net

## for CIFAR-10 by LYW : ResNet 19 Layers with only basic conv 3x3 in 12th_Sep_2016

def ResNet19_Conv3x3_basic_Cifar10(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 300 --> 150
        num_output=64, kernel_size=3, pad=1, stride=1,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )

    #net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2) # 150 --> 75 : in July 5th 2016 by LYW

    from_layer = 'conv1_relu'
    ResBasic33Body(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=True) # 75
    ResBasic33Body(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResBasic33Body(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    ResBasic33Body(net, 'res2c', '3a', out2a=128, out2b=128,  stride=2, use_branch1=True) # 75 --> 38
    ResBasic33Body(net, 'res3a', '3b', out2a=128, out2b=128,  stride=1, use_branch1=False) # 38
    ResBasic33Body(net, 'res3b', '3c', out2a=128, out2b=128,  stride=1, use_branch1=False) # 38

    from_layer = 'res3c'
    ResBasic33Body(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBasic33Body(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)
    ResBasic33Body(net, 'res4b', '4c', out2a=256, out2b=256,  stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4c_relu, pool=P.Pooling.AVE, global_pooling=True)

    net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net

## for CIFAR-10 by LYW : ResNet 19 Layers with only conv 3x3 in 14th_Sep_2016

def ResNet19_Conv3x3_l2_Cifar10(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 300 --> 150
        num_output=64, kernel_size=3, pad=1, stride=1,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )

    #net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2) # 150 --> 75 : in July 5th 2016 by LYW

    # use_branch1 means that projection shortcut is for increasing dimension.
    from_layer = 'conv1_relu'
    ResBasic333Body(net, from_layer, '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True) # 75
    ResBasic333Body(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    ResBasic333Body(net, 'res2b', '3a', out2a=256, out2b=256, out2c=256, stride=2, use_branch1=True) # 75 --> 38
    ResBasic333Body(net, 'res3a', '3d', out2a=256, out2b=256, out2c=256, stride=1, use_branch1=False) # 38

    from_layer = 'res3d'
    ResBasic333Body(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBasic333Body(net, 'res4a', '4b', out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4b_relu, pool=P.Pooling.AVE, global_pooling=True)

    net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net

## for CIFAR-10 by LYW : ResNet 19 Layers with only conv 3x3 in 14th_Sep_2016

def ResNet19_Conv3x3_k2_Cifar10(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 300 --> 150
        num_output=64, kernel_size=3, pad=1, stride=1,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )

    #net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2) # 150 --> 75 : in July 5th 2016 by LYW

    # use_branch1 means that projection shortcut is for increasing dimension.
    from_layer = 'conv1_relu'
    ResBasic333Body(net, from_layer, '2a', out2a=128, out2b=128, out2c=512, stride=1, use_branch1=True) # 75
    ResBasic333Body(net, 'res2a', '2b', out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)

    ResBasic333Body(net, 'res2b', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True) # 75 --> 38
    ResBasic333Body(net, 'res3a', '3d', out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False) # 38

    from_layer = 'res3d'
    ResBasic333Body(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBasic333Body(net, 'res4a', '4b', out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4b_relu, pool=P.Pooling.AVE, global_pooling=True)

    net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net


## for SSD by LYW : squeezeNet Layers with freeze layers in June_14_2016 
def SqueezeFire(net, from_layer, block_name, outS1, outE1, outE3):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  squeeze_postfix = 'squeeze1x1'

  conv_prefix = '{}/'.format(block_name) # 'fire2/'
  conv_postfix = squeeze_postfix

  convS_name = '{}{}'.format(conv_prefix,conv_postfix)
  reluS_name = '{}/relu_{}'.format(block_name,squeeze_postfix)


  #Squeeze
  # parameters for convolution layer with batchnorm.
  kwargs = {
      'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
      'weight_filler': dict(type='xavier'),
      'bias_filler': dict(type='constant', value=0) }

  net[convS_name] = L.Convolution(net[from_layer], num_output=outS1, kernel_size=1, stride=1, **kwargs)
  net[reluS_name] = L.ReLU(net[convS_name], in_place=True)

  #Expand
  merge_layers=[]
  expand_prefix = 'expand'
  expand_postfix_1 = '1x1'
  expand_postfix_3 = '3x3'
  
  convE1_name = '{}/{}{}'.format(block_name,expand_prefix,expand_postfix_1)
  convE3_name = '{}/{}{}'.format(block_name,expand_prefix,expand_postfix_3)

  reluE1_name = '{}/relu_{}{}'.format(block_name,expand_prefix,expand_postfix_1)
  reluE3_name = '{}/relu_{}{}'.format(block_name,expand_prefix,expand_postfix_3)

  # 1x1
  net[convE1_name] = L.Convolution(net[reluS_name], num_output=outE1, kernel_size=1, stride=1, **kwargs)
  net[reluE1_name] = L.ReLU(net[convE1_name], in_place=True)
  merge_layers.append(net[reluE1_name]) 

  # 3x3
  net[convE3_name] = L.Convolution(net[reluS_name], num_output=outE3, kernel_size=3, pad=1, stride=1, **kwargs)
  net[reluE3_name] = L.ReLU(net[convE3_name], in_place=True)
  merge_layers.append(net[reluE3_name]) 

  out_layer = '{}/concat'.format(block_name)
  net[out_layer] = L.Concat(*merge_layers, axis=1)


## for SSD by LYW : squeezeNet Layers with freeze layers in June_14_2016 
def SqueezeNetBodyWithResNet(net, from_layer, unFreeze_ConvLayers=[] ):
    
    kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)}

    relu_prefix = 'relu_'
    relu_name = '{}conv1'.format(relu_prefix)

    assert from_layer in net.keys()
    net.conv1 = L.Convolution(net[from_layer], num_output=96, kernel_size=7, pad=3, stride=2,**kwargs) # 300--> 150
    net[relu_name] = L.ReLU(net.conv1, in_place=True) 
    net.pool1 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=3, stride=2) # 150 --> 75


    SqueezeFire(net,'pool1','fire2',16,64,64) # output : fire2/concat
    SqueezeFire(net,'fire2/concat','fire3',16,64,64) #output : fire3/concat
    net['fire3_EltAdd'] = L.Eltwise(net['fire2/concat'], net['fire3/concat']) # residual


    SqueezeFire(net,'fire3_EltAdd','fire4',32,128,128) # output : fire4/concat
    net['pool4'] = L.Pooling(net['fire4/concat'], pool=P.Pooling.MAX, kernel_size=3, stride=2)   # 75 --> 38
    SqueezeFire(net,'pool4','fire5',32,128,128) #output : fire5/concat ( convLayer = fire5/squeeze1x1)
    net['fire5_EltAdd'] = L.Eltwise(net['pool4'], net['fire5/concat']) # residual


    SqueezeFire(net,'fire5_EltAdd','fire6',48,192,192) # output : fire6/concat
    SqueezeFire(net,'fire6/concat','fire7',48,192,192) #output : fire7/concat
    net['fire7_EltAdd'] = L.Eltwise(net['fire6/concat'], net['fire7/concat']) # residual
    SqueezeFire(net,'fire7_EltAdd','fire8',64,256,256) #output : fire8/concat  --> ## Conv layer : fire8/squeeze1x1

    net['pool8'] = L.Pooling(net['fire8/concat'], pool=P.Pooling.MAX, kernel_size=3, stride=2)  # 38 --> 19
    SqueezeFire(net,'pool8','fire9',64,256,256) #output : fire9/concat  --> ## Conv layer : fire9/squeeze1x1
    net['fire9_EltAdd'] = L.Eltwise(net['pool8'], net['fire9/concat']) # residual

    #net['fire9_EltAdd'] = L.Dropout(net['fire9_EltAdd'], dropout_ratio=0.5, in_place=True)
    


    # Update unfreeze layers.
    kwargs['param'] =  [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)]
    layers = net.keys()
    for unFreeze_layer in unFreeze_ConvLayers:
        if unFreeze_layer in layers:
            net.update(unFreeze_layer, kwargs)
    
    return net

#update commit

## for SSD by LYW : ResNet 19 Layers with freeze layers in July_16th_2016
# conv4_x added
def ResNet19_Conv4_BodyBiasFreeze(net, from_layer, use_pool5=False, noReduced=False, freeze_ConvLayers=[],freeze_ScaleLayers=[]):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 300 --> 150
        num_output=64, kernel_size=7, pad=3, stride=2,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )

    #net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2) # 150 --> 75
    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2) # 150 --> 75 : in July 5th 2016 by LYW

    ResBodyFreeze(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True) # 75
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)
    #ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    ResBody(net, 'res2b', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True) # 75 --> 38

    from_layer = 'res3a'
    # for i in xrange(1, 4):
    #   block_name = '3b{}'.format(i)
    #   ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
    #   from_layer = 'res{}'.format(block_name)
    #ResBody(net, from_layer, '3b', out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
    #ResBody(net, 'res3b', '3c', out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
    ResBody(net, 'res3a', '3d', out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False) # 38

    from_layer = 'res3d'
    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBody(net, 'res4a', '4b', out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      
    #added in July 4th 2016 by LYW
    #from_layer = 'res4a'
    #ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=True) # 19 : stride = 1
    
    if noReduced :

      ResBody(net, 'res4a', '4b', out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      ResBody(net, 'res4b', '4c', out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False) 
      ResBody(net, 'res4c', '4d', out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      ResBody(net, 'res4d', '4e', out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      ResBody(net, 'res4e', '4f', out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)

      # from_layer = 'res4f'
      # ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=2, use_branch1=True)
      ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)
      ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net

## by youngwan in 2016.09.20.
def BNReLUConvLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, eps=0.001,
    conv_prefix='conv_', conv_postfix='', bn_prefix='bn_', bn_postfix='',
    scale_prefix='scale_', scale_postfix='', bias_prefix='', bias_postfix='_bias'):
  if use_bn:
    #parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }


  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)


  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[from_layer], in_place=False, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  #if use_relu:
  relu_name = '{}_relu'.format(bn_name)
  net[relu_name] = L.ReLU(net[sb_name], in_place=True)

  
  from_layer = relu_name
  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  
#  if kernel_h == kernel_w:
  net[conv_name] = L.Convolution(net[relu_name], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  # else:
  #   net[conv_name] = L.Convolution(net[relu_name], num_output=num_output,
  #       kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
  #       stride_h=stride_h, stride_w=stride_w, **kwargs)
  # if dilation > 1:
  #   net.update(conv_name, {'dilation': dilation})


## by youngwan in 2016.09.20.
def ResPreActivBody(net, from_layer, block_name, out2a, out2b, stride, use_branch1):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'conv{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'proj'
    BNReLUConvLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2b, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
      branch1 = from_layer

  branch_name = 'branch2a'
  BNReLUConvLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  BNReLUConvLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
      num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2]) #layer saved!


## by youngwan in 2016.09.20.
def PreActiv_ResNet_15_Conv3x3_basic_l4(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    # BNReLUConvLayer(net, from_layer, '1', use_bn=True, use_relu=True, # 300 --> 150
    #     num_output=64, kernel_size=3, pad=1, stride=1,
    #     bn_prefix=bn_prefix, bn_postfix=bn_postfix,
    #   scale_prefix=scale_prefix, scale_postfix=scale_postfix
    # )
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    net['conv_1'] = L.Convolution(net[from_layer], num_output=64,
    kernel_size=3, pad=1, stride=1, **kwargs)	#32 



    from_layer = 'conv_1'
    ResPreActivBody(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=True) # 32
    ResPreActivBody(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResPreActivBody(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)


    ResPreActivBody(net, 'res2c', '3a', out2a=512, out2b=512,  stride=2, use_branch1=True) # 16
    ResPreActivBody(net, 'res3a', '3b', out2a=512, out2b=512,  stride=1, use_branch1=False) 


    from_layer = 'res3b'
    ResPreActivBody(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 8
    ResPreActivBody(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)

    if global_pool:
      net.pool5 = L.Pooling(net.res4b, pool=P.Pooling.AVE, global_pooling=True)

    net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net

## by youngwan in 2016.09.20.
def PreActiv_ResNet_15_Conv3x3_basic_l4_Cifar_100(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    # BNReLUConvLayer(net, from_layer, '1', use_bn=True, use_relu=True, # 300 --> 150
    #     num_output=64, kernel_size=3, pad=1, stride=1,
    #     bn_prefix=bn_prefix, bn_postfix=bn_postfix,
    #   scale_prefix=scale_prefix, scale_postfix=scale_postfix
    # )
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    net['conv_1'] = L.Convolution(net[from_layer], num_output=64,
    kernel_size=3, pad=1, stride=1, **kwargs) #32 



    from_layer = 'conv_1'
    ResPreActivBody(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=True) # 32
    ResPreActivBody(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResPreActivBody(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)


    ResPreActivBody(net, 'res2c', '3a', out2a=512, out2b=512,  stride=2, use_branch1=True) # 16
    ResPreActivBody(net, 'res3a', '3b', out2a=512, out2b=512,  stride=1, use_branch1=False) 


    from_layer = 'res3b'
    ResPreActivBody(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 8
    ResPreActivBody(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)

    if global_pool:
      net.pool5 = L.Pooling(net.res4b, pool=P.Pooling.AVE, global_pooling=True)

    net.fc100 = L.InnerProduct(net.pool5, num_output=100, weight_filler=dict(type='xavier'))

    return net


## by youngwan in 2016.09.22.
def PreActiv_ResNet_15_Conv3x3_basic_l4_SSD(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    # BNReLUConvLayer(net, from_layer, '1', use_bn=True, use_relu=True, # 300 --> 150
    #     num_output=64, kernel_size=3, pad=1, stride=1,
    #     bn_prefix=bn_prefix, bn_postfix=bn_postfix,
    #   scale_prefix=scale_prefix, scale_postfix=scale_postfix
    # )
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    net['conv_1'] = L.Convolution(net[from_layer], num_output=64,
    kernel_size=3, pad=1, stride=2, **kwargs)                     # 300 -->150



    from_layer = 'conv_1'
    ResPreActivBody(net, from_layer, '2a', out2a=64, out2b=64, stride=2, use_branch1=True) # 150 --> 70
    ResPreActivBody(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResPreActivBody(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)


    ResPreActivBody(net, 'res2c', '3a', out2a=512, out2b=512,  stride=2, use_branch1=True) # 70 --> 38
    ResPreActivBody(net, 'res3a', '3b', out2a=512, out2b=512,  stride=1, use_branch1=False) 


    from_layer = 'res3b'
    ResPreActivBody(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 38-->19
    ResPreActivBody(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)

    if global_pool:
      net.pool5 = L.Pooling(net.res4b, pool=P.Pooling.AVE, global_pooling=True)
      net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net

## by youngwan in 2016.09.22.
def PreActiv_ResNet_15_Conv3x3_basic_l3_SSD(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    # BNReLUConvLayer(net, from_layer, '1', use_bn=True, use_relu=True, # 300 --> 150
    #     num_output=64, kernel_size=3, pad=1, stride=1,
    #     bn_prefix=bn_prefix, bn_postfix=bn_postfix,
    #   scale_prefix=scale_prefix, scale_postfix=scale_postfix
    # )
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    net['conv_1'] = L.Convolution(net[from_layer], num_output=64,
    kernel_size=3, pad=1, stride=2, **kwargs)                     # 300 -->150



    from_layer = 'conv_1'
    ResPreActivBody(net, from_layer, '2a', out2a=64, out2b=64, stride=2, use_branch1=True) # 150 --> 70
    ResPreActivBody(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResPreActivBody(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)


    ResPreActivBody(net, 'res2c', '3a', out2a=384, out2b=384,  stride=2, use_branch1=True) # 70 --> 38
    ResPreActivBody(net, 'res3a', '3b', out2a=384, out2b=384,  stride=1, use_branch1=False) 


    from_layer = 'res3b'
    ResPreActivBody(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 38-->19
    ResPreActivBody(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)

    if global_pool:
      net.pool5 = L.Pooling(net.res4b, pool=P.Pooling.AVE, global_pooling=True)
      net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net

## by youngwan in 2016.09.21.

def RoR_15_Conv3x3_basic_l4(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    conv_prefix='conv_'

    # BNReLUConvLayer(net, from_layer, '1', use_bn=True, use_relu=True, # 300 --> 150
    #     num_output=64, kernel_size=3, pad=1, stride=1,
    #     bn_prefix=bn_prefix, bn_postfix=bn_postfix,
    #   scale_prefix=scale_prefix, scale_postfix=scale_postfix
    # )

    first_branch = 'conv_1'

    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    net[first_branch] = L.Convolution(net[from_layer], num_output=64,
    kernel_size=3, pad=1, stride=1, **kwargs)	#32


    ################################################################################################################################
    from_layer = first_branch
    branch_name = 'proj'
    dim = 64
    stride = 1
    BNReLUConvLayer(net, from_layer, branch_name, use_bn=True, use_relu=True, num_output=dim, kernel_size=1, pad=0, stride=stride )
    proj_branch = '{}{}'.format(conv_prefix, branch_name)


    ResPreActivBody(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=True) # 32
    ResPreActivBody(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResPreActivBody(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    main_branch = 'res2c'
    RoR_name = '{}_RoR'.format(main_branch)
    net[RoR_name] = L.Eltwise(net[proj_branch], net[main_branch])
    
    ################################################################################################################################
    branch_name = 'proj'
    from_layer = RoR_name
    dim = 512
    stride = 2
    BNReLUConvLayer(net, from_layer, branch_name, use_bn=True, use_relu=True, num_output=dim, kernel_size=1, pad=0, stride=stride )
    proj_branch = '{}{}'.format(conv_prefix, branch_name)
    
    ResPreActivBody(net, from_layer, '3a', out2a=512, out2b=512,  stride=2, use_branch1=True) # 16
    ResPreActivBody(net, 'res3a', '3b', out2a=512, out2b=512,  stride=1, use_branch1=False) 

    main_branch = 'res3b'
    RoR_name = '{}_RoR'.format(main_branch)
    net[RoR_name] = L.Eltwise(net[proj_branch], net[main_branch])

    ################################################################################################################################

    branch_name = 'proj'
    from_layer = RoR_name
    dim = 256
    stride = 2
    BNReLUConvLayer(net, from_layer, branch_name, use_bn=True, use_relu=True, num_output=dim, kernel_size=1, pad=0, stride=stride )
    proj_branch = '{}{}'.format(conv_prefix, branch_name)

    ResPreActivBody(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 8
    ResPreActivBody(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)

    main_branch = 'res4b'
    RoR_name = '{}_RoR'.format(main_branch)
    net[RoR_name] = L.Eltwise(net[proj_branch], net[main_branch])

    ################################################################################################################################

    branch_name = 'proj'
    from_layer = first_branch
    dim = 256
    stride = 4
    BNReLUConvLayer(net, from_layer, branch_name, use_bn=True, use_relu=True, num_output=dim, kernel_size=1, pad=0, stride=stride )
    proj_branch = '{}{}'.format(conv_prefix, branch_name)


    final_branch = RoR_name
    final_RoR_name = 'Final_RoR'
    net[final_RoR_name] = L.Eltwise(net[proj_branch], net[final_branch])


    ################################################################################################################################

    if global_pool:
      net.pool5 = L.Pooling(net[final_RoR_name], pool=P.Pooling.AVE, global_pooling=True)

    net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net


## by youngwan in 2016.09.21.

def RoR_15_Conv3x3_basic_l4_cifar_100(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    conv_prefix='conv_'

    # BNReLUConvLayer(net, from_layer, '1', use_bn=True, use_relu=True, # 300 --> 150
    #     num_output=64, kernel_size=3, pad=1, stride=1,
    #     bn_prefix=bn_prefix, bn_postfix=bn_postfix,
    #   scale_prefix=scale_prefix, scale_postfix=scale_postfix
    # )

    first_branch = 'conv_1'

    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    net[first_branch] = L.Convolution(net[from_layer], num_output=64,
    kernel_size=3, pad=1, stride=1, **kwargs) #32


    ################################################################################################################################
    from_layer = first_branch
    branch_name = 'proj'
    dim = 64
    stride = 1
    BNReLUConvLayer(net, from_layer, branch_name, use_bn=True, use_relu=True, num_output=dim, kernel_size=1, pad=0, stride=stride )
    proj_branch = '{}{}'.format(conv_prefix, branch_name)


    ResPreActivBody(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=True) # 32
    ResPreActivBody(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResPreActivBody(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    main_branch = 'res2c'
    RoR_name = '{}_RoR'.format(main_branch)
    net[RoR_name] = L.Eltwise(net[proj_branch], net[main_branch])
    
    ################################################################################################################################
    branch_name = 'proj'
    from_layer = RoR_name
    dim = 512
    stride = 2
    BNReLUConvLayer(net, from_layer, branch_name, use_bn=True, use_relu=True, num_output=dim, kernel_size=1, pad=0, stride=stride )
    proj_branch = '{}{}'.format(conv_prefix, branch_name)
    
    ResPreActivBody(net, from_layer, '3a', out2a=512, out2b=512,  stride=2, use_branch1=True) # 16
    ResPreActivBody(net, 'res3a', '3b', out2a=512, out2b=512,  stride=1, use_branch1=False) 

    main_branch = 'res3b'
    RoR_name = '{}_RoR'.format(main_branch)
    net[RoR_name] = L.Eltwise(net[proj_branch], net[main_branch])

    ################################################################################################################################

    branch_name = 'proj'
    from_layer = RoR_name
    dim = 256
    stride = 2
    BNReLUConvLayer(net, from_layer, branch_name, use_bn=True, use_relu=True, num_output=dim, kernel_size=1, pad=0, stride=stride )
    proj_branch = '{}{}'.format(conv_prefix, branch_name)

    ResPreActivBody(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 8
    ResPreActivBody(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)

    main_branch = 'res4b'
    RoR_name = '{}_RoR'.format(main_branch)
    net[RoR_name] = L.Eltwise(net[proj_branch], net[main_branch])

    ################################################################################################################################

    branch_name = 'proj'
    from_layer = first_branch
    dim = 256
    stride = 4
    BNReLUConvLayer(net, from_layer, branch_name, use_bn=True, use_relu=True, num_output=dim, kernel_size=1, pad=0, stride=stride )
    proj_branch = '{}{}'.format(conv_prefix, branch_name)


    final_branch = RoR_name
    final_RoR_name = 'Final_RoR'
    net[final_RoR_name] = L.Eltwise(net[proj_branch], net[final_branch])


    ################################################################################################################################

    if global_pool:
      net.pool5 = L.Pooling(net[final_RoR_name], pool=P.Pooling.AVE, global_pooling=True)

    net.fc100 = L.InnerProduct(net.pool5, num_output=100, weight_filler=dict(type='xavier'))

    return net 




def Inception_ResBody(net, from_layer, block_name, out2a, out2b_1,out2b_3,out2c_1,out2c_3,out_merge, stride, use_branch1):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out_merge, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  branch_name = 'branch2a'
  out_name = '{}_1x1'.format(branch_name)
  ConvBNLayer(net, from_layer, out_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  branch2 = '{}{}'.format(conv_prefix, out_name)

  branch_name = 'branch2b'
  out_name = '{}_1x1'.format(branch_name)
  ConvBNLayer(net, from_layer, out_name, use_bn=True, use_relu=True,
      num_output=out2b_1, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)

  prev_layer = '{}{}'.format(conv_prefix, out_name)
  out_name = '{}_3x3'.format(branch_name)
  
  ConvBNLayer(net, prev_layer, out_name, use_bn=True, use_relu=True,
      num_output=out2b_3, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)

  branch3 = '{}{}'.format(conv_prefix, out_name)



  branch_name = 'branch2c'
  out_name = '{}_1x1'.format(branch_name)

  ConvBNLayer(net, from_layer, out_name, use_bn=True, use_relu=True,
      num_output=out2c_1, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)

  from_layer = '{}{}'.format(conv_prefix, out_name)
  out_name = '{}_3x3_a'.format(branch_name)

  ConvBNLayer(net, from_layer, out_name, use_bn=True, use_relu=True,
      num_output=out2c_3, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)

  from_layer = '{}{}'.format(conv_prefix, out_name)
  out_name = '{}_3x3_b'.format(branch_name)

  ConvBNLayer(net, from_layer, out_name, use_bn=True, use_relu=True,
      num_output=out2c_3, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)

  branch4 = '{}{}'.format(conv_prefix, out_name)



  merge_layers=[]
  merge_layers.append(net[branch2])
  merge_layers.append(net[branch3])
  merge_layers.append(net[branch4])

  out_layer = '{}/concat'.format(block_name)
  net[out_layer] = L.Concat(*merge_layers, axis=1)

  from_layer = out_layer
  out_name = '{}_1x1'.format(out_layer)

  ConvBNLayer(net, from_layer, out_name, use_bn=True, use_relu=True,
      num_output=out_merge, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)

  merge_branch = '{}{}'.format(conv_prefix, out_name)



  res_name = 'inception_{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[merge_branch]) #layer saved!
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)






def Inception_Res_Conv3x3_basic_Cifar10(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 30 --> 30
        num_output=64, kernel_size=3, pad=1, stride=1,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )


    from_layer = 'conv1_relu'
    ResBasic33Body(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=False) # 75
    ResBasic33Body(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResBasic33Body(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    ResBasic33Body(net, 'res2c', '3a', out2a=128, out2b=128,  stride=2, use_branch1=True) # 7
    #ResBasic33Body(net, 'res3a', '3a', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38
    Inception_ResBody(net,'res3a','3a',out2a=64, out2b_1=64,out2b_3=128,out2c_1=64,out2c_3=128,out_merge=128,stride=1,use_branch1=False)
    

    from_layer = 'inception_3a'
    ResBasic33Body(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 38 --> 19 : stride = 2    
    ResBasic33Body(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)
    #ResBasic33Body(net, 'res4b', '4c', out2a=256, out2b=256,  stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4b_relu, pool=P.Pooling.AVE, global_pooling=True)

    net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net


def Inception_Res_Conv3x3_basic_l2_Cifar10(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 32 --> 32
        num_output=64, kernel_size=3, pad=1, stride=1,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )


    from_layer = 'conv1_relu'
    ResBasic33Body(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=False) # 32
    ResBasic33Body(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResBasic33Body(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    ResBasic33Body(net, 'res2c', '3a', out2a=256, out2b=256,  stride=2, use_branch1=True) # 32-->16
    #ResBasic33Body(net, 'res3a', '3a', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38
    Inception_ResBody(net,'res3a','3a',out2a=64, out2b_1=64,out2b_3=256,out2c_1=64,out2c_3=256,out_merge=256,stride=1,use_branch1=False)
    

    from_layer = 'inception_3a'
    ResBasic33Body(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 16 --> 8   
    ResBasic33Body(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)
    #ResBasic33Body(net, 'res4b', '4c', out2a=256, out2b=256,  stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4b_relu, pool=P.Pooling.AVE, global_pooling=True)

    net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net


def Inception_Res_v2_Cifar10(net, from_layer, global_pool=True):
    

    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayerWithBias(net, from_layer, 'conv1', use_bn=True, use_relu=True, # 32 --> 32
        num_output=64, kernel_size=3, pad=1, stride=1,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix
    )


    from_layer = 'conv1_relu'
    ResBasic33Body(net, from_layer, '2a', out2a=64, out2b=64, stride=1, use_branch1=False) # 32
    ResBasic33Body(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)
    ResBasic33Body(net, 'res2b', '2c', out2a=64, out2b=64, stride=1, use_branch1=False)

    #ResBasic33Body(net, 'res2c', '3a', out2a=256, out2b=256,  stride=2, use_branch1=True) # 32-->16
    #ResBasic33Body(net, 'res3a', '3a', out2a=512, out2b=512,  stride=1, use_branch1=False) # 38
    Inception_ResBody(net,'res2c','3a',out2a=64, out2b_1=64,out2b_3=256,out2c_1=64,out2c_3=256,out_merge=256,stride=2,use_branch1=True) # 32--> 16
    from_layer = 'inception_3a'
    Inception_ResBody(net,from_layer,'3b',out2a=64, out2b_1=64,out2b_3=256,out2c_1=64,out2c_3=256,out_merge=256,stride=1,use_branch1=False)
    

    from_layer = 'inception_3b'
    ResBasic33Body(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True) # 16 --> 8   
    ResBasic33Body(net, 'res4a', '4b', out2a=256, out2b=256,  stride=1, use_branch1=False)
    #ResBasic33Body(net, 'res4b', '4c', out2a=256, out2b=256,  stride=1, use_branch1=False)
      
    if global_pool:
      net.pool5 = L.Pooling(net.res4b_relu, pool=P.Pooling.AVE, global_pooling=True)

    net.fc10 = L.InnerProduct(net.pool5, num_output=10, weight_filler=dict(type='xavier'))

    return net
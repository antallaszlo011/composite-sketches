import pandas as pd
import numpy as np
from fastai.vision import *
import matplotlib.pyplot as plt

if __name__=='__main__':
    # pd.set_option('display.max_columns', 500)

    path = Path('DATA\\faces\\')
    # print(path)

    ## Function to filter validation samples
    def validation_func(x):
        return 'validation' in x

    tfms = get_transforms(do_flip=False, flip_vert=False, max_rotate=30, max_lighting=0.3)
    # print(tfms)

    src = (ImageList.from_csv(path, csv_name='labels.csv')
        .split_by_valid_func(validation_func)
        .label_from_df(cols='tags',label_delim=' '))

    data = (src.transform(tfms, size=128)
        .databunch(bs=32).normalize(imagenet_stats))

    print(data.c, '\n', data.classes)

    # plt.figure()
    # data.show_batch(rows=2, figsize=(20,12))
    # plt.savefig('batch.png')

    # arch = models.resnet50

    # acc_02 = partial(accuracy_thresh, thresh=0.2)
    # acc_03 = partial(accuracy_thresh, thresh=0.3)
    # acc_04 = partial(accuracy_thresh, thresh=0.4)
    # acc_05 = partial(accuracy_thresh, thresh=0.5)
    # f_score = partial(fbeta, thresh=0.2)
    # learn = cnn_learner(data, arch, metrics=[acc_02, acc_03, acc_04, acc_05, f_score])

    # learn.lr_find()

    # plt.clf()
    # learn.recorder.plot()
    # plt.savefig('learning_rate_1.png')

    # lr = 1e-2

    # learn.fit_one_cycle(1, slice(lr))

    # learn.fit_one_cycle(4, slice(lr))

    # learn.save('ff_stage-1-rn50')
    # learn.load('ff_stage-1-rn50')

    # learn.unfreeze()

    # learn.lr_find()

    # plt.clf()
    # learn.recorder.plot()
    # plt.savefig('learning_rate_2.png')

    # learn.fit_one_cycle(5, slice(1e-5, lr/5))

    # learn.save('ff_stage-2-rn50')
    # learn.load('ff_stage-2-rn50')

    data = (src.transform(tfms, size=256)
           .databunch(bs=8).normalize(imagenet_stats))

    acc_05 = partial(accuracy_thresh, thresh=0.5)
    f_score = partial(fbeta, thresh=0.5)
    learn = cnn_learner(data, models.resnet50, pretrained=False,metrics=[acc_05, f_score])
    learn.load("ff_stage-2-rn50")

    learn.freeze()

    learn.lr_find()
    learn.recorder.plot()

    lr = 0.01

    learn.fit_one_cycle(1, slice(lr))

    learn.save('ff_stage-1-256-rn50')

    learn = cnn_learner(data, models.resnet50, pretrained=False)
    learn.load("ff_stage-1-256-rn50")

    m = learn.model.eval()
    print(m)

    idx=1
    x,y = data.valid_ds[idx]
    plt.figure()
    x.show()
    plt.savefig('x_show.png')
    print(data.valid_ds.y[idx])

    xb, _ = data.one_item(x)
    xb_im = Image(data.denorm(xb)[0])
    xb = xb.cuda()

    from fastai.callbacks.hooks import *

    def hooked_backward(cat=y):
        with hook_output(m[0]) as hook_a: 
            with hook_output(m[0], grad=True) as hook_g:
                preds = m(xb)
                #preds[0,str(data.valid_ds.y[idx])].backward()
        return hook_a,hook_g

    hook_a,hook_g = hooked_backward()



    acts  = hook_a.stored[0].cpu()
    acts.shape


    avg_acts = acts.mean(0)
    avg_acts.shape

    def show_heatmap(hm):
        _,ax = plt.subplots()
        xb_im.show(ax)
        ax.imshow(hm, alpha=0.6, extent=(0,256,256,0),
                  interpolation='bilinear', cmap='magma');

    avg_acts

    show_heatmap(avg_acts)

    idx=700
    x,y = data.valid_ds[idx]
    xb, _ = data.one_item(x)
    xb_im = Image(data.denorm(xb)[0])
    xb = xb.cuda()

    hook_a,hook_g = hooked_backward()
    acts  = hook_a.stored[0].cpu()
    avg_acts = acts.mean(0)
    show_heatmap(avg_acts)

    avg_acts

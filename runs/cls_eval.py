if __name__ == '__main__':
    import numpy as np
    from datasets.ISIC2018 import *
    from models import backbone
    from keras.models import Model, Input
    from keras.layers import Average
    from  misc_utils.visualization_utils import BatchVisualization
    from  misc_utils.eval_utils import get_confusion_matrix, get_precision_recall
    from numpy import array
    from numpy import argmax
    from sklearn.metrics import accuracy_score

    backbone_name = 'densenet201'

    # This number can be changed from 0 to 4 to evaluate a specific fold's performance
    k_fold = 3

    version = '0'

    run_name = 'task3_' + backbone_name + '_k' + str(k_fold) + '_v' + version

    _, (x, y_true), _ = load_training_data(task_idx=3,
                                           output_size=224,
                                           idx_partition=k_fold)

    model = backbone(backbone_name).classification_model(load_from=run_name)

    # max_num_images = 32

    max_num_images = x.shape[0]

    x = x[:max_num_images]

    y_true = y_true[:max_num_images]

    y_pred = model.predict(x)

    _ = get_confusion_matrix(y_true=y_true, y_pred=y_pred, print_cm=True)

    get_precision_recall(y_true=y_true, y_pred=y_pred)

    bv = BatchVisualization(images=x,

                            true_labels=y_true,

                            pred_labels=y_pred)

    bv()


    # backbone_name = 'densenet201'
    #
    # k_fold = 0
    # version = '0'
    #
    # run_name_1 = 'task3_' + backbone_name + '_k' + str(k_fold) + '_v' + version
    # run_name_2 = 'task3_' + 'nasnet' + '_k' + str(k_fold) + '_v' + version
    #
    #
    # _, (x, y_true), _ = load_training_data(task_idx=3,
    #
    #                                        output_size=224,
    #
    #                                        idx_partition=k_fold)
    # print(x.shape)
    # input_shape = x[0, :, :, :].shape
    # model_input = Input(shape=input_shape)
    #
    # model_densenet = backbone(backbone_name).classification_model(load_from=run_name_1)
    # model_nasnet = backbone('nasnet').classification_model(load_from=run_name_2)
    #
    #
    #
    # model_densenet.load_weights('model_data/task3_densenet201_k1_v0/task3_densenet201_k1_v0.hdf5')
    # model_nasnet.load_weights('model_data/task3_nasnet_k0_v0/task3_nasnet_k0_v0.hdf5')
    #
    #
    #
    #
    # models = [model_densenet,model_nasnet]
    # # # max_num_images = 32
    # #
    # # def ensemble(models, model_input):
    # #     outputs = [model.outputs[0] for model in models]
    # #     y = Average()(outputs)
    # #     model = Model(model_input, y, name='ensemble')
    # #     return model
    # #
    # #
    #
    # # ensemble_model = ensemble(models, model_input)
    #
    # max_num_images = x.shape[0]
    # x = x[:max_num_images]
    # y_true = y_true[:max_num_images]
    #
    # outputs = [model.outputs[0] for model in models]
    # y = Average()(outputs)
    #
    # model = Model(model_input, y, name='ensemble')
    #
    # y_pred = model.predict(x)
    # make predictions
    # yhats = [model.predict(x) for model in models]
    #
    # y = Average()(yhats)



    #yhats = array(yhats)
    # sum across ensembles
    #summed = numpy.sum(yhats, axis=0)
    # argmax across classes
    #outcomes = argmax(summed, axis=1)


    #print(outcomes)

    #
    #
    #
    # y_pred = model_densenet.predict(x)
    #
    #
    #
    # _ = get_confusion_matrix(y_true=y_true, y_pred=y_pred, print_cm=True)
    #
    # get_precision_recall(y_true=y_true, y_pred=y_pred)

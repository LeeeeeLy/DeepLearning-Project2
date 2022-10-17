import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
import sys
from keras import backend as K
from os import listdir
from os.path import isfile, join

#get data from files
def GetData(path, data_type, TestorTrainorValid):
    #get to the folder
    if TestorTrainorValid == 'test':
        datapath = path+'/Testing'
    elif TestorTrainorValid == "train":
        datapath = path+'/Training'
    else:
        datapath = path+'/Validation'
            
    #read in all file names in that folder    
    filenames = [f for f in listdir(datapath) if isfile(join(datapath, f))]
    #print(filenames)
    certain_type = []
    #corrspond command line parameters to filename
    if data_type == 'DIA':
        datatype = '_BP Dia_mmHg'
    elif data_type == 'mmHg':
        datatype = '_BP_mmHg'
    elif data_type == 'mean':
        datatype = '_LA Mean BP_mmHg'
    elif data_type == 'EDA':
        datatype = '_EDA_microsiemens'
    elif data_type == 'sys':
        datatype = '_LA Systolic BP_mmHg'
    elif data_type == 'pulse':
        datatype = '_Pulse Rate_BPM'
    elif data_type == 'volt':
        datatype = '_Resp_Volts'
    elif data_type == 'resp':
        datatype = '_Respiration Rate_BPM'
    else:#for the case 'all'
        datatype = '_'

    for match in filenames:
        if datatype in match:
            certain_type.append(match)
    #store values in files as data and classes for labels
    dataset = []
    labels = []
    for txtfile in certain_type:
        names_file = open(datapath + '/' + txtfile, 'r')
        dataset.append(np.loadtxt(names_file))
        if txtfile[6].isdigit():#for the case class=10
            labels.append(int(txtfile[5])+int(txtfile[6]))
        else:#for class 1-9
           labels.append(int(txtfile[5]))
           
    #normalization
#     somemax = []
#     for i in range(1,len(dataset)):
#         somemax.append(dataset[i].max())
#     
#     dataset = dataset/max(somemax)    

    return dataset, labels

def MakeModel():
    
    n_classes = 10
    #timesteps for LSTM
    timestep = 1000
    learningrate = 0.001
    numbatchs = 32
    #create sequential model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(n_classes, numbatchs, 
    input_length=timestep))
    model.add(tf.keras.layers.Conv1D(64, kernel_size=3, input_shape=(n_classes, numbatchs), activation="relu"))
    
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
    #model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learningrate), metrics=['accuracy'])
    return model

def Predict(model,path,data_type):
    (x_test, y_test) = GetData(path, data_type, 'test')
    #pad data with 0 (make sure same length)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=1000)
    y_test = np.array(y_test)
    y_test = to_categorical(y_test)
    test_eval = model.evaluate(x_test, y_test, verbose=0)
    accuracy = test_eval[1]
    #convert labels back
    y_test=np.argmax(y_test, axis=1)
    y_test[1]
    #predict and format output to use with sklearn
    predict = model.predict(x_test)
    #print(np.shape(predict))
    predict = np.argmax(predict, axis=1)
    #print(np.shape(predict))
    #macro precision and recall
    precisionMacro = precision_score(y_test, predict, average='macro')
    recallMacro = recall_score(y_test, predict, average='macro')
    #micro precision and recall
    precisionMicro = precision_score(y_test, predict, average='micro')
    recallMicro = recall_score(y_test, predict, average='micro')
    #Confusion Matrix 
    confMat = confusion_matrix(y_test, predict)
    #F1 scores
    F1Macro = 2*((precisionMacro*recallMacro)/(precisionMacro+recallMacro+K.epsilon()))
    F1Micro = 2*((precisionMicro*recallMicro)/(precisionMicro+recallMicro+K.epsilon()))
    print("Test accuracy: ", accuracy)
    print("Macro precision: ", precisionMacro)
    print("Micro precision: ", precisionMicro)
    print("Macro recall: ", recallMacro)
    print("Micro recall: ", recallMicro)
    print("Macro F1: ", F1Macro)
    print("Micro F1: ", F1Micro)
    print(confMat)   
    Metrics = [accuracy,recallMicro,recallMacro,precisionMicro,precisionMacro,F1Micro,F1Macro,confMat]
    return Metrics

def Train(name,path,data_type):
    numbatchs = 16
    numepochs = 20
    (x_train, y_train) = GetData(path, data_type, 'train')    
    (x_validation, y_validation) = GetData(path, data_type, 'validation')
    #pad data with 0 (make sure same length)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=1000)
    x_validation = tf.keras.preprocessing.sequence.pad_sequences(x_validation, maxlen=1000)
    #format data so we can use it
    y_train = np.array(y_train)     
    y_validation = np.array(y_validation)    
    y_train = to_categorical(y_train)
    y_validation = to_categorical(y_validation)
    model = MakeModel()
    model.fit(x_train, y_train,
               batch_size=numbatchs,
               epochs=numepochs,
               validation_data=[x_validation, y_validation])
    model.save("./models/"+name+".h5")
    print("Model saved.")
    
def TrainBest(name,path,data_type):
    numbatchs = 16
    numepochs = 20
    checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath="./models/"+name+".h5", 
                             monitor='val_accuracy',
                             verbose=1, 
                             save_best_only=True,
                             mode='max'),
                #tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.01, patience=2)
                  ]
                
    (x_train, y_train) = GetData(path, data_type, 'train')    
    (x_validation, y_validation) = GetData(path, data_type, 'validation')
    #pad data with 0 (make sure same length)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=1000)
    x_validation = tf.keras.preprocessing.sequence.pad_sequences(x_validation, maxlen=1000)
    #format data so we can use it
    y_train = np.array(y_train)     
    y_validation = np.array(y_validation)    
    y_train = to_categorical(y_train)
    y_validation = to_categorical(y_validation)
    model = MakeModel()
    model.fit(x_train, y_train,
               batch_size=numbatchs,
               epochs=numepochs,
               validation_data=[x_validation, y_validation], callbacks=checkpoint)

def Test(name,path,data_type):
    print("Loading Test Data")
    (x_test, y_test) = GetData(path, data_type, 'test')
    print("Loading model")
    model = tf.keras.models.load_model("./models/"+name+".h5")
    print("Making predictions on test data")
    Predict(model,path,data_type)
    
    

def main():
    TestorTrain = sys.argv[1]
    path = sys.argv[2]
    name = sys.argv[3]
    data_type = sys.argv[4]
    if TestorTrain == "train":
        Train(name,path,data_type)
    elif sys.argv[1] == "trainBest":
        TrainBest(name,path,data_type)
    elif sys.argv[1] == "test":
        Test(name,path,data_type)

if __name__ == "__main__":
    main()
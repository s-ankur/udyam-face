import numpy as np
import models
from dataset import kaggle

def fit(train_faces,train_emotions,test_faces,test_emotions)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit_generator(data_generator.flow(train_faces, train_emotions,batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=val_data)

def save_model()
    json_string=model.to_json()
    with open(json_path,'w') as json_file:
        json_file.write(json_string)

if __name__ =="__main__":
    model=cnn_model(input_shape,len(emotions))
    training_data, training_labels = kaggle.load_data('train')
    prediction_data, prediction_labels = kaggle.load_data('test')
    fit(training_data, training_labels,prediction_data, prediction_labels)
    save_model()
from CNN_classifiers.cnn_classifier import CNN
data_path = '../DR_data/characters_numbers' # so the path being passed to the class when initialized should be the one relative to that script
model_path =  'saved_models'
classifier = CNN(data_path=data_path, save_model_path=model_path, num_classes=30)
# 30 - 9 numbers, 21 letters
classes = classifier.train_model(type='characters')
print(classes)
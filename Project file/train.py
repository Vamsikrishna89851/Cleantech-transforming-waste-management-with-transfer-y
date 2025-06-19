import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import argparse
from preprocess import get_data_generators

def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(include_top=False,
                                                   input_shape=(224,224,3),
                                                   weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train transfer learning model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to image dataset')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    train_gen, val_gen = get_data_generators(args.data_dir)
    model = build_model(train_gen.num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=args.epochs)

    model.save('waste_classifier.h5')
    print("Model trained and saved as 'waste_classifier.h5'")
import config
import os
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import joblib


def inference_model():
    """Returns the model that will be used for inference"""
    # print(os.getcwd(), "\n")
    with open(os.path.join(config.save_model_path, 'tokenizer' + str(config.num_decoder_tokens)), 'rb') as file:
        tokenizer = joblib.load(file)
    # loading encoder model. This remains the same
    inf_encoder_model = load_model(os.path.join(config.save_model_path, 'encoder_model.h5'))
    print("\n Encoder weights loaded successfully \n")

    # inference decoder model loading
    decoder_inputs = Input(shape=(None, config.num_decoder_tokens))
    decoder_dense = Dense(config.num_decoder_tokens, activation='softmax')
    decoder_lstm = LSTM(config.latent_dim, return_sequences=True, return_state=True)
    decoder_state_input_h = Input(shape=(config.latent_dim,))
    decoder_state_input_c = Input(shape=(config.latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    inf_decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    inf_decoder_model.load_weights(os.path.join(config.save_model_path, 'decoder_model_weights.h5'))

    opt = keras.optimizers.Adam(lr=0.0003)
    # model.compile(metrics=['accuracy'], optimizer=opt, loss='categorical_crossentropy')
    inf_decoder_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n Decoder weights loaded successfully \n")
    return tokenizer, inf_encoder_model, inf_decoder_model


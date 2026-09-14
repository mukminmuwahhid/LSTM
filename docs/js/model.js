// Reconstructs the trained Keras Sequential LSTM in TensorFlow.js from the
// plain-JSON weight export (docs/model/weights.json) produced by
// src/export_web.py. Built by hand (units/activation/weights only) instead
// of via the tfjs-converter CLI, since the layers here are simple enough
// that this avoids an extra Python toolchain + version-matrix headache.
async function loadTfjsModel(weightsUrl) {
  const res = await fetch(weightsUrl);
  const spec = await res.json();

  const model = tf.sequential();
  const built = [];
  let first = true;

  for (const layer of spec.layers) {
    if (layer.class_name === 'LSTM') {
      // Keras defaults recurrentActivation to 'sigmoid'; tfjs defaults to
      // 'hardSigmoid' — must be set explicitly or predictions drift.
      const opts = {
        units: layer.units,
        returnSequences: layer.return_sequences,
        activation: 'tanh',
        recurrentActivation: 'sigmoid',
      };
      if (first) { opts.inputShape = spec.input_shape; first = false; }
      const l = tf.layers.lstm(opts);
      model.add(l);
      built.push({ spec: layer, layer: l });
    } else if (layer.class_name === 'Dense') {
      const l = tf.layers.dense({ units: layer.units, activation: layer.activation });
      model.add(l);
      built.push({ spec: layer, layer: l });
    }
    // Dropout: no trainable weights, no-op at inference — skip.
  }

  for (const { spec: layerSpec, layer } of built) {
    const tensors = layerSpec.weights.map((w) => tf.tensor(w));
    layer.setWeights(tensors);
    tensors.forEach((t) => t.dispose());
  }

  return model;
}

if (typeof module !== 'undefined') module.exports = { loadTfjsModel };

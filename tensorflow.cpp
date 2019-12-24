/*************************************************************************/
/*  tensorflow.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "tensorflow.h"

#include <tensorflow/lite/builtin_op_data.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/stderr_reporter.h>
#include <tensorflow/lite/string_util.h>

#include <algorithm>
#include <queue>

#include "core/bind/core_bind.h"

extern bool input_floating;
template <class T>
void resize(T *out, uint8_t *in, int image_height, int image_width,
		int image_channels, int wanted_height, int wanted_width,
		int wanted_channels, bool floating) {
	int number_of_pixels = image_height * image_width * image_channels;
	std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);

	int base_index = 0;

	// two inputs: input and new_sizes
	interpreter->AddTensors(2, &base_index);
	// one output
	interpreter->AddTensors(1, &base_index);
	// set input and output tensors
	interpreter->SetInputs({ 0, 1 });
	interpreter->SetOutputs({ 2 });

	// set parameters of tensors
	TfLiteQuantizationParams quant = TfLiteQuantizationParams();
	interpreter->SetTensorParametersReadWrite(
			0, kTfLiteFloat32, "input",
			{ 1, image_height, image_width, image_channels }, quant);
	interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", { 2 },
			quant);
	interpreter->SetTensorParametersReadWrite(
			2, kTfLiteFloat32, "output",
			{ 1, wanted_height, wanted_width, wanted_channels }, quant);

	tflite::ops::builtin::BuiltinOpResolver resolver;
	const TfLiteRegistration *resize_op =
			resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
	auto *params = reinterpret_cast<TfLiteResizeBilinearParams *>(
			malloc(sizeof(TfLiteResizeBilinearParams)));
	params->align_corners = false;
	interpreter->AddNodeWithParameters({ 0, 1 }, { 2 }, nullptr, 0, params, resize_op,
			nullptr);

	interpreter->AllocateTensors();

	// fill input image
	// in[] are integers, cannot do memcpy() directly
	auto input = interpreter->typed_tensor<float>(0);
	for (int i = 0; i < number_of_pixels; i++) {
		input[i] = in[i];
	}

	// fill new_sizes
	interpreter->typed_tensor<int>(1)[0] = wanted_height;
	interpreter->typed_tensor<int>(1)[1] = wanted_width;

	interpreter->Invoke();

	auto output = interpreter->typed_tensor<float>(2);
	auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

	for (int i = 0; i < output_number_of_pixels; i++) {
		if (floating)
			//out[i] = (output[i] - s->input_mean) / s->input_std;
			out[i] = output[i];
		else
			out[i] = (uint8_t)output[i];
	}
}

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
template <class T>
void get_top_n(T *prediction, int prediction_size, size_t num_results,
		float threshold, std::vector<std::pair<float, int> > *top_results,
		bool input_floating) {
	// Will contain top N results in ascending order.
	std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
			std::greater<std::pair<float, int> > >
			top_result_pq;

	const long count = prediction_size; // NOLINT(runtime/int)
	for (int i = 0; i < count; ++i) {
		float value;
		if (input_floating)
			value = prediction[i];
		else
			value = prediction[i] / 255.0;
		// Only add it if it beats the threshold and has a chance at being in
		// the top N.
		if (value < threshold) {
			continue;
		}

		top_result_pq.push(std::pair<float, int>(value, i));

		// If at capacity, kick the smallest value out.
		if (top_result_pq.size() > num_results) {
			top_result_pq.pop();
		}
	}

	// Copy to output vector and reverse into descending order.
	while (!top_result_pq.empty()) {
		top_results->push_back(top_result_pq.top());
		top_result_pq.pop();
	}
	std::reverse(top_results->begin(), top_results->end());
}

template <class T>
void get_top_n(T *prediction, int prediction_size, size_t num_results,
		float threshold, std::vector<std::pair<float, int> > *top_results,
		bool input_floating);

// explicit instantiation so that we can use them otherwhere
template void get_top_n<uint8_t>(uint8_t *, int, size_t,
		float, std::vector<std::pair<float, int> > *,
		bool);
template void get_top_n<float>(float *, int, size_t,
		float, std::vector<std::pair<float, int> > *,
		bool);

template <class T>
void resize(T *out, uint8_t *in, int image_height, int image_width,
		int image_channels, int wanted_height, int wanted_width,
		int wanted_channels, bool floating);

// explicit instantiation
template void resize<uint8_t>(uint8_t *, uint8_t *, int, int,
		int, int, int,
		int, bool);
template void resize<float>(float *, uint8_t *, int, int,
		int, int, int,
		int, bool);

void TensorflowAiInstance::set_labels(PoolStringArray p_string) {
	labels = p_string;
}

PoolStringArray TensorflowAiInstance::get_labels() {
	return labels;
}

void TensorflowAiInstance::set_texture(Ref<Texture> p_texture) {
	texture = p_texture;
}

Ref<Texture> TensorflowAiInstance::get_texture() {
	return texture;
}

void TensorflowAiInstance::set_tensorflow_model(const Ref<TensorflowModel> &p_model) {
	tensorflow_model = p_model;
	if (tensorflow_model.is_valid()) {
		tensorflow_model->register_owner(this);
	}
}

Ref<TensorflowModel> TensorflowAiInstance::get_tensorflow_model() const {
	return tensorflow_model;
}

void TensorflowAiInstance::inference() {
	ERR_FAIL_COND(!interpreter);

	// Run inference
	ERR_FAIL_COND(interpreter->Invoke() != kTfLiteOk);
	tflite::PrintInterpreterState(interpreter.get());
}

void TensorflowAiInstance::_bind_methods() {
	ClassDB::bind_method(D_METHOD("inference"), &TensorflowAiInstance::inference);
	ClassDB::bind_method(D_METHOD("allocate_tensor_buffers"), &TensorflowAiInstance::allocate_tensor_buffers);
	ClassDB::bind_method(D_METHOD("set_tensorflow_model", "model"), &TensorflowAiInstance::set_tensorflow_model);
	ClassDB::bind_method(D_METHOD("get_tensorflow_model"), &TensorflowAiInstance::get_tensorflow_model);
	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &TensorflowAiInstance::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &TensorflowAiInstance::get_texture);
	ClassDB::bind_method(D_METHOD("set_labels", "label"), &TensorflowAiInstance::set_labels);
	ClassDB::bind_method(D_METHOD("get_labels"), &TensorflowAiInstance::get_labels);
	ClassDB::bind_method(D_METHOD("set_label_path", "label"), &TensorflowAiInstance::set_label_path);
	ClassDB::bind_method(D_METHOD("get_label_path"), &TensorflowAiInstance::get_label_path);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "label_path"), "set_label_path", "get_label_path");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_STRING_ARRAY, "labels",PROPERTY_HINT_NONE, "", PROPERTY_USAGE_INTERNAL), "set_labels", "get_labels");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "tensorflow_model", PROPERTY_HINT_RESOURCE_TYPE, "TensorflowModel"), "set_tensorflow_model", "get_tensorflow_model");
}
void TensorflowAiInstance::set_label_path(String p_path) {
	label_path = p_path;
}
String TensorflowAiInstance::get_label_path() const {
	return label_path;
}
void TensorflowAiInstance::_notification(int p_notification) {
	if (p_notification == Node::NOTIFICATION_READY && !Engine::get_singleton()->is_editor_hint()) {
		Ref<_File> file;
		file.instance();
		file->open(get_label_path(), _File::READ);
		PoolStringArray l;
		while (!file->eof_reached()) {
			l.push_back(file->get_line());
		}
		set_labels(l);
		allocate_tensor_buffers();
	}
}

TensorflowAiInstance::TensorflowAiInstance() {
	model = NULL;
	interpreter = NULL;
}

void TensorflowAiInstance::allocate_tensor_buffers() {
	ERR_FAIL_COND(texture.is_null());
	Ref<Image> img = texture->get_data();
	PoolVector<uint8_t> model_data = tensorflow_model->get_data();
	PoolVector<uint8_t>::Read r = model_data.read();
	model = tflite::FlatBufferModel::BuildFromBuffer((const char *)r.ptr(), model_data.size());
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*model, resolver);
	builder(&interpreter);
	ERR_FAIL_COND(interpreter == NULL);

	interpreter->UseNNAPI(true);
	interpreter->SetAllowFp16PrecisionForFp32(true);

	print_verbose("Tensors size: " + itos(interpreter->tensors_size()));
	print_verbose("Nodes size: " + itos(interpreter->nodes_size()));
	print_verbose("Inputs: " + itos(interpreter->inputs().size()));
	ERR_FAIL_COND(interpreter->inputs().size() == 0);
	print_verbose("Input(0) name: " + String(interpreter->GetInputName(0)));

	int32_t t_size = interpreter->tensors_size();
	for (int32_t i = 0; i < t_size; i++) {
		if (interpreter->tensor(i)->name) {
			print_verbose(itos(i) + ": " + String(interpreter->tensor(i)->name) + ", " +
						  itos(interpreter->tensor(i)->bytes) + ", " +
						  itos(interpreter->tensor(i)->type) + ", " +
						  itos(interpreter->tensor(i)->params.scale) + ", " +
						  itos(interpreter->tensor(i)->params.zero_point));
		}
	}
	interpreter->SetNumThreads(_OS::get_singleton()->get_processor_count());

	int32_t input = interpreter->inputs()[0];
	print_verbose("input:");

	const std::vector<int> inputs = interpreter->inputs();
	const std::vector<int> outputs = interpreter->outputs();

	print_verbose("number of inputs: " + itos(inputs.size()));
	print_verbose("number of outputs: " + itos(outputs.size()));

	if (interpreter->AllocateTensors() != kTfLiteOk) {
		ERR_FAIL_MSG("Tensorflow can't allocate tensors");
	}

	TfLiteIntArray *dims = interpreter->tensor(input)->dims;
	// get input dimension from the input tensor metadata
	// assuming one input only
	int32_t wanted_height = dims->data[1];
	int32_t wanted_width = dims->data[2];
	int32_t wanted_channels = dims->data[3];
	wanted_height = real_t(wanted_width) * real_t(img->get_height()) / real_t(img->get_width());
	img->resize(wanted_width, wanted_height);

	if (wanted_channels == 3) {
		img->convert(Image::FORMAT_RGB8);
	} else if (wanted_channels == 4) {
		img->convert(Image::FORMAT_RGBA8);
	} else {
		ERR_FAIL_MSG("Tensorflow: invalid image format");
	}
	switch (interpreter->tensor(input)->type) {
		case kTfLiteFloat32: {
			PoolVector<uint8_t>::Write float_write = img->get_data().write();
			resize<float>(interpreter->typed_tensor<float>(input), float_write.ptr(),
					img->get_height(), img->get_width(), wanted_channels, wanted_height,
					wanted_width, wanted_channels, true);
			break;
		}
		case kTfLiteUInt8: {
			PoolVector<uint8_t>::Write uint_8_write = img->get_data().write();
			resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), uint_8_write.ptr(),
					img->get_height(), img->get_width(), wanted_channels, wanted_height,
					wanted_width, wanted_channels, false);
			break;
		}
		default: {
			ERR_FAIL_MSG("Tensorflow: cannot handle input type " + itos(interpreter->tensor(input)->type) + " yet");
		}
	}

	if (interpreter->Invoke() != kTfLiteOk) {
		ERR_FAIL_MSG("Tensorflow can't invoke");
	}

	const float threshold = 0.001f;
	const int32_t number_of_results = 10;

	std::vector<std::pair<float, int> > top_results;

	int output = interpreter->outputs()[0];
	TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
	// assume output dims to be something like (1, 1, ... ,size)
	size_t output_size = output_dims->data[output_dims->size - 1];
	switch (interpreter->tensor(output)->type) {
		case kTfLiteFloat32:
			get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
					number_of_results, threshold, &top_results, true);
			break;
		case kTfLiteUInt8:
			get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
					output_size, number_of_results, threshold,
					&top_results, false);
			break;
		default:
			ERR_FAIL_MSG("Tensorflow: cannot handle input type " + itos(interpreter->tensor(input)->type) + " yet");
	}

	for (const auto &result : top_results) {
		const float confidence = result.first;
		const int index = result.second;
		if (labels.get(index) != String("")) {
			print_line(rtos(confidence) + ": " + itos(index) + " " + labels[index]);
		}
	}
}

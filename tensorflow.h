/*************************************************************************/
/*  tensorflow.h                                                         */
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

#ifndef TENSORFLOW_H
#define TENSORFLOW_H

#include "core/reference.h"
#include "loader_tflite.h"
#include "scene/main/node.h"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/stderr_reporter.h>

class AiInstance : public Node {
	GDCLASS(AiInstance, Node);
	virtual void inference() = 0;
};

class TensorflowAiInstance : public AiInstance {
	GDCLASS(TensorflowAiInstance, AiInstance);

	String label_path;
protected:
	static void _bind_methods();
	std::unique_ptr<tflite::FlatBufferModel> model;
	std::unique_ptr<tflite::Interpreter> interpreter;
	Ref<TensorflowModel> tensorflow_model;
	Ref<Texture> texture;
	PoolStringArray labels;
	void _notification(int p_notification);

public:
	void set_label_path(String p_path);
	String get_label_path() const;
	void set_labels(PoolStringArray p_string);
	PoolStringArray get_labels();
	void set_texture(Ref<Texture> p_texture);
	Ref<Texture> get_texture();
	void set_tensorflow_model(const Ref<TensorflowModel> &p_model);
	Ref<TensorflowModel> get_tensorflow_model() const;
	void inference();
	TensorflowAiInstance();
	void allocate_tensor_buffers();
};

#endif

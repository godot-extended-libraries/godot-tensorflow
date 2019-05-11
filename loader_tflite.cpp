/*************************************************************************/
/*  loader_tflite.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "loader_tflite.h"
#include "core/io/resource_importer.h"
#include "core/io/resource_saver.h"
#include "core/os/file_access.h"

void TensorflowModel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_data", "data"), &TensorflowModel::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &TensorflowModel::get_data);
	ADD_PROPERTY(PropertyInfo(Variant::POOL_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_data", "get_data");
}

void TensorflowModel::set_data(const PoolVector<uint8_t> &p_data) {
	if (!data.empty()) {
		data.clear();
	}

	int datalen = p_data.size();
	if (datalen) {

		PoolVector<uint8_t>::Read r = p_data.read();
		data.resize(p_data.size());
		uint8_t *dataptr = data.ptrw();
		copymem(dataptr, r.ptr(), p_data.size());
	}
}

PoolVector<uint8_t> TensorflowModel::get_data() const {
	PoolVector<uint8_t> pv;

	if (!data.empty()) {
		pv.resize(data.size());
		{

			PoolVector<uint8_t>::Write w = pv.write();
			copymem(w.ptr(), data.ptr(), data.size());
		}
	}

	return pv;
}

Error TensorflowModel::load_model(FileAccess *f) {
	Vector<uint8_t> raw_data;
	size_t length = f->get_len();
	ERR_FAIL_COND_V(length == 0, ERR_FILE_CORRUPT);
	raw_data.resize(length);
	f->get_buffer(raw_data.ptrw(), length);
	if (raw_data[4] != 'T' || raw_data[5] != 'F' || raw_data[6] != 'L' || raw_data[7] != '3') {
		ERR_FAIL_V(ERR_FILE_UNRECOGNIZED);
	}
	data.clear();
	data.resize(length);
	copymem(data.ptrw(), raw_data.ptr(), raw_data.size());
	return OK;
}

RES TensorflowModelResourceLoader::load(const String &p_path, const String &p_original_path, Error *r_error) {
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	if (!f) {
		return ERR_FILE_CANT_OPEN;
	}
	Ref<TensorflowModel> lib;
	lib.instance();
	Error err = lib->load_model(f);
	ERR_FAIL_COND_V(err != OK, ERR_FILE_CORRUPT);
	return lib;
}

void TensorflowModelResourceLoader::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("tflite");
}

bool TensorflowModelResourceLoader::handles_type(const String &p_type) const {
	return p_type == "TensorflowModel";
}

String TensorflowModelResourceLoader::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "tflite")
		return "TensorflowModel";
	return "";
}

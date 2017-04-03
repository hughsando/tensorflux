/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <hx/CffiPrime.h>


#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/cc/framework/cc_op_gen.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/framework/types.pb_text.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/cc/framework/cc_op_gen.h"

namespace tensorflow {

namespace {

const int kRightMargin = 79;

// Converts:
//   bazel-out/.../genfiles/(external/YYY/)?XX
// to: XX.
string GetPath(const string& dot_h_fname) {
  auto pos = dot_h_fname.find("/genfiles/");
  string result = dot_h_fname;
  if (pos != string::npos) {
    // - 1 account for the terminating null character (\0) in "/genfiles/".
    result = dot_h_fname.substr(pos + sizeof("/genfiles/") - 1);
  }
  if (result.size() > sizeof("external/") &&
      result.compare(0, sizeof("external/") - 1, "external/") == 0) {
    result = result.substr(sizeof("external/") - 1);
    pos = result.find("/");
    if (pos != string::npos) {
      result = result.substr(pos + 1);
    }
  }
  return result;
}

// Converts: some/path/to/file.xx
// to: file
// (note that suffix is removed)
string GetFilename(const string& path) {
  size_t slash_pos = path.rfind('/');
  if (slash_pos == path.npos) slash_pos = -1;
  size_t dot_pos = path.rfind('.');
  return path.substr(slash_pos + 1, dot_pos - (slash_pos + 1));
}

// Converts:
//   cc/ops/gen_foo_ops.h
// to:
//   CC_OPS_GEN_FOO_OPS_H_
string ToGuard(const string& path) {
  string guard;
  guard.reserve(path.size() + 1);  // + 1 -> trailing _
  for (const char c : path) {
    if (c >= 'A' && c <= 'Z') {
      guard += c;
    } else if (c >= 'a' && c <= 'z') {
      guard += c + 'A' - 'a';
    } else {
      guard += '_';
    }
  }
  guard += '_';
  return guard;
}

// Converts: some_name_xyz
// to: Some Name Xyz
string ToTitle(const string& name) {
  string title = name;
  for (int i = 0; i < title.size(); ++i) {
    if (title[i] == '_') title[i] = ' ';
  }
  str_util::TitlecaseString(&title, " ");
  return title;
}

// Change:     Into:
//   ABC         /// ABC
//               ///
//   DEF         /// DEF
string MakeComment(StringPiece text, StringPiece indent) {
  string ret;
  while (!text.empty()) {
    int last_non_space = -1;
    int newline;
    for (newline = 0; newline < static_cast<int>(text.size()); ++newline) {
      if (text[newline] == '\n') break;
      if (text[newline] != ' ') last_non_space = newline;
    }
    if (last_non_space == -1) {
      strings::StrAppend(&ret, indent, "///\n");
    } else {
      strings::StrAppend(&ret, indent, "/// ",
                         text.substr(0, last_non_space + 1), "\n");
    }
    text.remove_prefix(newline + 1);
  }
  return ret;
}

string PrintString(const string& str) {
  return strings::StrCat("\"", str_util::CEscape(str), "\"");
}

string PrintTensorShape(const TensorShape& shape) {
  string ret = "{";
  for (int d = 0; d < shape.dims(); ++d) {
    if (d > 0) strings::StrAppend(&ret, ", ");
    strings::StrAppend(&ret, shape.dim_size(d));
  }
  strings::StrAppend(&ret, "}");
  return ret;
}

template <typename T>
string PrintArray(int64 num_elts, const T* array) {
  string ret;
  for (int64 i = 0; i < num_elts; ++i) {
    if (i > 0) strings::StrAppend(&ret, ", ");
    strings::StrAppend(&ret, array[i]);
  }
  return ret;
}

string PrintTensor(const TensorProto& tensor_proto) {
  Tensor t(tensor_proto.dtype());
  CHECK(t.FromProto(tensor_proto));
  const int64 num_elts = t.NumElements();
  switch (t.dtype()) {
    case DT_FLOAT:
      return PrintArray(num_elts, t.flat<float>().data());
    case DT_DOUBLE:
      return PrintArray(num_elts, t.flat<double>().data());
    case DT_INT32:
      return PrintArray(num_elts, t.flat<int32>().data());
    case DT_UINT8:
    case DT_QUINT8:
      return PrintArray(num_elts, t.flat<uint8>().data());
    case DT_UINT16:
    case DT_QUINT16:
      return PrintArray(num_elts, t.flat<uint16>().data());
    case DT_INT16:
    case DT_QINT16:
      return PrintArray(num_elts, t.flat<int16>().data());
    case DT_INT8:
    case DT_QINT8:
      return PrintArray(num_elts, t.flat<int8>().data());
    case DT_INT64:
      return PrintArray(num_elts, t.flat<int64>().data());
    case DT_BOOL:
      return PrintArray(num_elts, t.flat<bool>().data());
    case DT_STRING: {
      string ret;
      for (int64 i = 0; i < num_elts; ++i) {
        if (i > 0) strings::StrAppend(&ret, " ");
        strings::StrAppend(&ret, str_util::CEscape(t.flat<string>()(i)));
      }
      return ret;
    }
    default: {
      LOG(FATAL) << "Not handling type " << EnumName_DataType(t.dtype());
      return string();
    }
  }
}

string PrintAttrValue(string op, const AttrValue& attr_value) {
  switch (attr_value.value_case()) {
    case AttrValue::kS:
      return PrintString(attr_value.s());
    case AttrValue::kI:
      return strings::StrCat(attr_value.i());
    case AttrValue::kF: {
      const float f = attr_value.f();
      return strings::StrCat(attr_value.f(), floorf(f) == f ? ".0" : "", "f");
    }
    case AttrValue::kB:
      return attr_value.b() ? "true" : "false";
    case AttrValue::kType:
      return EnumName_DataType(attr_value.type());
    case AttrValue::kShape:
      return PrintTensorShape(TensorShape(attr_value.shape()));
    case AttrValue::kTensor:
      return strings::StrCat(
          "Input::Initializer(", "{", PrintTensor(attr_value.tensor()), "}, ",
          PrintTensorShape(TensorShape(attr_value.tensor().tensor_shape())),
          ").AsTensorProto()");
    case AttrValue::kList: {
      string ret = "{";
      if (attr_value.list().s_size() > 0) {
        for (int i = 0; i < attr_value.list().s_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, PrintString(attr_value.list().s(i)));
        }
      } else if (attr_value.list().i_size() > 0) {
        for (int i = 0; i < attr_value.list().i_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, attr_value.list().i(i));
        }
      } else if (attr_value.list().f_size() > 0) {
        for (int i = 0; i < attr_value.list().f_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          const float f = attr_value.list().f(i);
          strings::StrAppend(&ret, f, floorf(f) == f ? ".0" : "", "f");
        }
      } else if (attr_value.list().b_size() > 0) {
        for (int i = 0; i < attr_value.list().b_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, attr_value.list().b(i) ? "true" : "false");
        }
      } else if (attr_value.list().type_size() > 0) {
        for (int i = 0; i < attr_value.list().type_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret,
                             EnumName_DataType(attr_value.list().type(i)));
        }
      } else if (attr_value.list().shape_size() > 0) {
        for (int i = 0; i < attr_value.list().shape_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(
              &ret, PrintTensorShape(TensorShape(attr_value.list().shape(i))));
        }
      }
      strings::StrAppend(&ret, "}");
      return ret;
    }
    default:
      LOG(FATAL) << "Unsupported Attr type: " << op << " "
                 << attr_value.value_case();
  }
  return "<Unknown AttrValue type>";  // Prevent missing return warning
}

string ToCamelCase(const string& str) {
  string result;
  const char joiner = '_';
  size_t i = 0;
  bool cap = true;
  while (i < str.size()) {
    const char c = str[i++];
    if (c == joiner) {
      cap = true;
    } else if (cap) {
      result += toupper(c);
      cap = false;
    } else {
      result += c;
    }
  }
  return result;
}

// Returns a <string, bool> pair. The string is the C++ type name to be used for
// attr_type when defining an object of that type. The bool is a flag to
// indicate whether to treat the type as const when accepting the C++ type as an
// argument to a function.
std::pair<const char*, bool> AttrTypeName(StringPiece attr_type) {
  static const std::unordered_map<StringPiece, std::pair<const char*, bool>,
                                  StringPiece::Hasher>
      attr_type_map{
          {"string", {"StringPiece", false}},
          {"list(string)", {"gtl::ArraySlice<string>", true}},
          {"int", {"int64", false}},
          {"list(int)", {"gtl::ArraySlice<int>", true}},
          {"float", {"float", false}},
          {"list(float)", {"gtl::ArraySlice<float>", true}},
          {"bool", {"bool", false}},
          {"list(bool)", {"gtl::ArraySlice<bool>", true}},
          {"type", {"DataType", false}},
          {"list(type)", {"DataTypeSlice", true}},
          {"shape", {"TensorShape", false}},
          {"list(shape)", {"gtl::ArraySlice<TensorShape>", true}},
          {"tensor", {"TensorProto", true}},
          {"list(tensor)", {"gtl::ArraySlice<TensorProto>", true}},
          {"func", {"NameAttrList", true}},
      };

  auto entry = attr_type_map.find(attr_type);
  if (entry == attr_type_map.end()) {
    LOG(FATAL) << "Unsupported Attr type: " << attr_type;
    return {"", false};
  }
  return entry->second;
}

bool IsCPPKeyword(StringPiece name) {
  static const std::unordered_set<StringPiece, StringPiece::Hasher>
      // Keywords obtained from http://en.cppreference.com/w/cpp/keyword
      kCPPReserved{
          "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel",
          "atomic_commit", "atomic_noexcept", "auto", "bitand", "bitor", "bool",
          "break", "case", "catch", "char", "char16_t", "char32_t", "class",
          "compl", "concept", "const", "const_cast", "constexpr", "continue",
          "decltype", "default", "delete", "do", "double", "dynamic_cast",
          "else", "enum", "explicit", "export", "extern", "false", "final",
          "float", "for", "friend", "goto", "if", "import", "inline", "int",
          "long", "module", "mutable", "namespace", "new", "noexcept", "not",
          "not_eq", "nullptr", "operator", "or", "or_eq", "override", "private",
          "protected", "public", "register", "reinterpret_cast", "requires",
          "return", "short", "signed", "sizeof", "static", "static_assert",
          "static_cast", "struct", "switch", "synchronized", "template", "this",
          "thread_local", "throw", "true", "try", "typedef", "typeid",
          "typename", "union", "unsigned", "using", "virtual", "void",
          "volatile", "wchar_t", "while", "xor", "xor_eq",

          // The following are not C++ keywords, but names of local variables
          // and parameters used in the op constructor. Treating them as
          // keywords, so that other parameter names don't conflict with these.
          "builder", "node", "ret", "scope", "unique_name",
      };
  return kCPPReserved.count(name) > 0;
}

string AvoidCPPKeywords(StringPiece name) {
  if (IsCPPKeyword(name)) {
    return strings::StrCat(name, "_");
  }
  return name.ToString();
}

void InferArgAttributes(const OpDef::ArgDef& arg,
                        std::unordered_map<string, string>* inferred_attrs) {
  if (!arg.type_attr().empty()) {
    gtl::InsertIfNotPresent(inferred_attrs, arg.type_attr(), arg.name());
  } else if (!arg.type_list_attr().empty()) {
    gtl::InsertIfNotPresent(inferred_attrs, arg.type_list_attr(), arg.name());
  }
  if (!arg.number_attr().empty()) {
    gtl::InsertIfNotPresent(inferred_attrs, arg.number_attr(), arg.name());
  }
}

void InferOpAttributes(
    const OpDef& op_def,
    std::unordered_map<string, string>* inferred_input_attrs) {
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const auto& arg(op_def.input_arg(i));
    InferArgAttributes(arg, inferred_input_attrs);
  }
}

bool ArgIsList(const OpDef::ArgDef& arg) {
  return !arg.type_list_attr().empty() || !arg.number_attr().empty();
}

bool HasOptionalAttrs(
    const OpDef& op_def,
    const std::unordered_map<string, string>& inferred_input_attrs) {
  for (int i = 0; i < op_def.attr_size(); ++i) {
    const auto& attr(op_def.attr(i));
    if ((inferred_input_attrs.find(attr.name()) ==
         inferred_input_attrs.end()) &&
        attr.has_default_value()) {
      return true;
    }
  }
  return false;
}

struct OpInfo {
  // graph_op_def: The OpDef used by the runtime, has the names that
  //   must be used when calling NodeBuilder.
  // interface_op_def: The OpDef used in the interface in the generated
  //   code, with possibly overridden names and defaults.
  explicit OpInfo(const OpDef& graph_op_def, const OpDef& inteface_op_def,
                  const std::vector<string>& aliases);
  string GetOpAttrStruct() const;
  string GetConstructorDecl(StringPiece op_name_prefix,
                            bool include_attr) const;
  void WriteClassDecl(WritableFile* h) const;
  void GetOutput(string* out) const;
  string GetConstructorBody() const;
  void WriteClassDef(WritableFile* cc) const;

  string op_name;
  std::vector<string> arg_types;
  std::vector<string> arg_names;
  std::vector<string> output_types;
  std::vector<string> output_names;
  std::vector<bool> is_list_output;
  bool has_optional_attrs;
  string comment;

  const OpDef& graph_op_def;
  const OpDef& op_def;
  const std::vector<string>& aliases;
  std::unordered_map<string, string> inferred_input_attrs;
};

OpInfo::OpInfo(const OpDef& g_op_def, const OpDef& i_op_def,
               const std::vector<string>& a)
    : graph_op_def(g_op_def), op_def(i_op_def), aliases(a) {
  op_name = op_def.name();
  InferOpAttributes(op_def, &inferred_input_attrs);
  has_optional_attrs = HasOptionalAttrs(op_def, inferred_input_attrs);
  arg_types.push_back("const ::tensorflow::Scope&");
  arg_names.push_back("scope");

  if (op_def.has_deprecation()) {
    if (!op_def.summary().empty()) {
      comment = strings::StrCat(op_def.summary(), "\n");
    }
    strings::StrAppend(&comment, "DEPRECATED at GraphDef version ",
                       op_def.deprecation().version(), ":\n",
                       op_def.deprecation().explanation(), ".\n");
  } else if (op_def.summary().empty()) {
    comment = "TODO: add doc.\n";
  } else {
    comment = strings::StrCat(op_def.summary(), "\n");
  }
  if (!op_def.description().empty()) {
    strings::StrAppend(&comment, "\n", op_def.description(), "\n");
  }
  strings::StrAppend(&comment, "\nArguments:\n* scope: A Scope object\n");

  // Process inputs
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const auto& arg(op_def.input_arg(i));
    arg_types.push_back(strings::StrCat(
        "::tensorflow::", ArgIsList(arg) ? "InputList" : "Input"));
    arg_names.push_back(AvoidCPPKeywords(arg.name()));

    // TODO(keveman): Include input type information.
    StringPiece description = arg.description();
    if (!description.empty()) {
      ConsumeEquals(&description);
      strings::StrAppend(&comment, "* ", AvoidCPPKeywords(arg.name()), ": ",
                         arg.description(), "\n");
    }
  }

  // Process attrs
  string required_attrs_comment;
  string optional_attrs_comment;
  for (int i = 0; i < op_def.attr_size(); ++i) {
    const auto& attr(op_def.attr(i));
    // Skip inferred arguments
    if (inferred_input_attrs.count(attr.name()) > 0) continue;

    const auto entry = AttrTypeName(attr.type());
    const auto attr_type_name = entry.first;
    const bool use_const = entry.second;
    string attr_name = AvoidCPPKeywords(attr.name());

    string attr_comment;
    if (!attr.description().empty()) {
      // TODO(keveman): Word wrap and indent this, to handle multi-line
      // descriptions.
      strings::StrAppend(&attr_comment, "* ", attr_name, ": ",
                         attr.description(), "\n");
    }
    if (attr.has_default_value()) {
      strings::StrAppend(&optional_attrs_comment, attr_comment);
    } else {
      strings::StrAppend(&required_attrs_comment, attr_comment);
      arg_types.push_back(strings::StrCat(
          use_const ? "const " : "", attr_type_name, use_const ? "&" : ""));
      arg_names.push_back(attr_name);
    }
  }

  strings::StrAppend(&comment, required_attrs_comment);

  if (!optional_attrs_comment.empty()) {
    strings::StrAppend(&comment, "\nOptional attributes (see `Attrs`):\n");
    strings::StrAppend(&comment, optional_attrs_comment);
  }

  // Process outputs
  for (int i = 0; i < op_def.output_arg_size(); ++i) {
    const auto& arg = op_def.output_arg(i);
    bool is_list = ArgIsList(arg);
    output_types.push_back(
        strings::StrCat("::tensorflow::", is_list ? "OutputList" : "Output"));
    output_names.push_back(AvoidCPPKeywords(arg.name()));
    is_list_output.push_back(is_list);
  }

  strings::StrAppend(&comment, "\nReturns:\n");
  if (op_def.output_arg_size() == 0) {  // No outputs.
    strings::StrAppend(&comment, "* the created `Operation`\n");
  } else if (op_def.output_arg_size() == 1) {  // One output
    if (is_list_output[0]) {
      strings::StrAppend(&comment, "* `OutputList`: ");
    } else {
      strings::StrAppend(&comment, "* `Output`: ");
    }
    if (op_def.output_arg(0).description().empty()) {
      strings::StrAppend(&comment, "The ", op_def.output_arg(0).name(),
                         " tensor.\n");
    } else {
      // TODO(josh11b): Word wrap this.
      strings::StrAppend(&comment, op_def.output_arg(0).description(), "\n");
    }
  } else {  // Multiple outputs.
    for (int i = 0; i < op_def.output_arg_size(); ++i) {
      if (is_list_output[i]) {
        strings::StrAppend(&comment, "* `OutputList`");
      } else {
        strings::StrAppend(&comment, "* `Output`");
      }
      strings::StrAppend(&comment, " ", output_names[i]);
      if (op_def.output_arg(i).description().empty()) {
        strings::StrAppend(&comment, "\n");
      } else {
        // TODO(josh11b): Word wrap this.
        strings::StrAppend(&comment, ": ", op_def.output_arg(i).description(),
                           "\n");
      }
    }
  }

  if (!aliases.empty()) {
    strings::StrAppend(&comment, "\nAliases:\n");
    for (const auto& alias : aliases) {
      strings::StrAppend(&comment, "* ", alias, "\n");
    }
  }
  comment = MakeComment(comment, "");
}

string OpInfo::GetOpAttrStruct() const {
  string struct_fields;
  string setters;

  for (int i = 0; i < op_def.attr_size(); ++i) {
    const auto& attr(op_def.attr(i));
    // If attr will be inferred or it doesn't have a default value, don't
    // add it to the struct.
    if ((inferred_input_attrs.find(attr.name()) !=
         inferred_input_attrs.end()) ||
        !attr.has_default_value()) {
      continue;
    }
    const auto entry = AttrTypeName(attr.type());
    const auto attr_type_name = entry.first;
    const bool use_const = entry.second;
    const string camel_case_name = ToCamelCase(attr.name());
    const string suffix =
        (camel_case_name == op_name || camel_case_name == "Attrs") ? "_" : "";
    const string attr_func_def =
        strings::StrCat(camel_case_name, suffix, "(", use_const ? "const " : "",
                        attr_type_name, use_const ? "&" : "");

    string attr_comment;
    if (!attr.description().empty()) {
      strings::StrAppend(&attr_comment, attr.description(), "\n\n");
    }
    strings::StrAppend(&attr_comment, "Defaults to ",
                       SummarizeAttrValue(attr.default_value()), "\n");
    attr_comment = MakeComment(attr_comment, "    ");

    strings::StrAppend(&setters, attr_comment);
    strings::StrAppend(&setters, "    Attrs ", attr_func_def, " x) {\n");
    strings::StrAppend(&setters, "      Attrs ret = *this;\n");
    strings::StrAppend(&setters, "      ret.", attr.name(), "_ = x;\n");
    strings::StrAppend(&setters, "      return ret;\n    }\n\n");

    strings::StrAppend(
        &struct_fields, "    ", attr_type_name, " ", attr.name(), "_ = ",
        PrintAttrValue(op_def.name(), attr.default_value()), ";\n");
  }

  if (struct_fields.empty()) {
    return "";
  }

  string attrs_comment =
      strings::StrCat("Optional attribute setters for ", op_name, "\n");
  string struct_decl = MakeComment(attrs_comment, "  ");
  strings::StrAppend(&struct_decl, "  struct Attrs {\n");
  strings::StrAppend(&struct_decl, setters, struct_fields);
  strings::StrAppend(&struct_decl, "  };\n");

  return struct_decl;
}

string OpInfo::GetConstructorDecl(StringPiece op_name_prefix,
                                  bool include_attr) const {
  const string prefix = strings::StrCat(op_name_prefix, op_name, "(");
  string c_decl;
  for (int i = 0; i < arg_types.size(); ++i) {
    if (i > 0) strings::StrAppend(&c_decl, ", ");
    strings::StrAppend(&c_decl, arg_types[i], " ", arg_names[i]);
  }
  if (include_attr && has_optional_attrs) {
    strings::StrAppend(&c_decl, ", const ", op_name, "::Attrs& attrs");
  }
  strings::StrAppend(&c_decl, ")");
  return WordWrap(prefix, c_decl, kRightMargin);
}

void OpInfo::WriteClassDecl(WritableFile* h) const {
  string class_decl = comment;
  strings::StrAppend(&class_decl, "class ", op_name, " {\n");
  strings::StrAppend(&class_decl, " public:\n");
  if (has_optional_attrs) {
    strings::StrAppend(&class_decl, GetOpAttrStruct());
  }
  strings::StrAppend(&class_decl, "  ",
                     GetConstructorDecl("", /* include_attr */ false), ";\n");
  if (has_optional_attrs) {
    strings::StrAppend(&class_decl, "  ",
                       GetConstructorDecl("", /* include_attr */ true), ";\n");
  }
  if (output_types.empty()) {
    // Allow casting this class to Operation.
    strings::StrAppend(&class_decl,
                       "  operator ::tensorflow::Operation() const { "
                       "return operation; }\n");
  } else if (output_types.size() == 1) {
    if (is_list_output[0]) {
      // Write the subscript operator, allowing out[i] for the list-typed
      // output.
      strings::StrAppend(&class_decl,
                         "  ::tensorflow::Output operator[](size_t index) "
                         "const { return ",
                         output_names[0], "[index]; }\n\n");

    } else {
      // Write type cast functions, allowing casting this class to Input and
      // Output.
      strings::StrAppend(&class_decl,
                         "  operator ::tensorflow::Output() const { return ",
                         output_names[0], "; }\n");
      strings::StrAppend(&class_decl,
                         "  operator ::tensorflow::Input() const { return ",
                         output_names[0], "; }\n");
      // Write node() to get the Node* directly.
      strings::StrAppend(&class_decl,
                         "  ::tensorflow::Node* node() const { return ",
                         output_names[0], ".node(); }\n");
    }
  }
  // Add the static functions to set optional attrs
  if (has_optional_attrs) {
    strings::StrAppend(&class_decl, "\n");
    for (int i = 0; i < op_def.attr_size(); ++i) {
      const auto& attr(op_def.attr(i));
      if ((inferred_input_attrs.find(attr.name()) !=
           inferred_input_attrs.end()) ||
          !attr.has_default_value()) {
        continue;
      }
      const auto entry = AttrTypeName(attr.type());
      const auto attr_type_name = entry.first;
      const bool use_const = entry.second;
      const string camel_case_name = ToCamelCase(attr.name());
      const string suffix =
          (camel_case_name == op_name || camel_case_name == "Attrs") ? "_" : "";
      const string attr_func_def = strings::StrCat(
          camel_case_name, suffix, "(", use_const ? "const " : "",
          attr_type_name, use_const ? "&" : "");
      strings::StrAppend(&class_decl, "  static Attrs ", attr_func_def,
                         " x) {\n");
      strings::StrAppend(&class_decl, "    return Attrs().", camel_case_name,
                         suffix, "(x);\n");
      strings::StrAppend(&class_decl, "  }\n");
    }
  }

  strings::StrAppend(&class_decl, "\n");

  if (output_types.empty()) {
    strings::StrAppend(&class_decl, "  Operation operation;\n");
  }
  for (int i = 0; i < output_types.size(); ++i) {
    strings::StrAppend(&class_decl, "  ", output_types[i], " ", output_names[i],
                       ";\n");
  }

  strings::StrAppend(&class_decl, "};\n");
  if (!aliases.empty()) {
    for (const auto& alias : aliases) {
      strings::StrAppend(&class_decl, "typedef ", op_name, " ", alias, ";\n");
    }
  }
  strings::StrAppend(&class_decl, "\n");
  TF_CHECK_OK(h->Append(class_decl));
}

void OpInfo::GetOutput(string* out) const {
  const string scope_str = arg_names[0];
  string return_on_error =
      strings::StrCat("if (!", scope_str, ".ok()) return;");

  // No outputs.
  if (op_def.output_arg_size() == 0) {
    strings::StrAppend(out, "  this->operation = Operation(ret);\n  return;\n");
    return;
  }
  if (op_def.output_arg_size() == 1) {
    // One output, no need for NameRangeMap
    if (is_list_output[0]) {
      strings::StrAppend(out,
                         "  for (int64 i = 0; i < ret->num_outputs(); ++i)\n");
      strings::StrAppend(out, "    this->", output_names[0],
                         ".push_back(Output(ret, i));\n");
    } else {
      strings::StrAppend(out, "  this->", output_names[0],
                         " = Output(ret, 0);\n");
    }
    return;
  }
  strings::StrAppend(out, "  ::tensorflow::NameRangeMap _outputs_range;\n");
  strings::StrAppend(
      out,
      "  ::tensorflow::Status _status_ = "
      "::tensorflow::NameRangesForNode(ret->def(), ret->op_def(), "
      "nullptr, &_outputs_range);\n");
  strings::StrAppend(out, "  if (!_status_.ok()) {\n", "    ", scope_str,
                     ".UpdateStatus(_status_);\n", "    return;\n");
  strings::StrAppend(out, "  }\n\n");

  for (int i = 0; i < op_def.output_arg_size(); ++i) {
    const string arg_range = strings::StrCat(
        "_outputs_range[\"", graph_op_def.output_arg(i).name(), "\"]");
    if (is_list_output[i]) {
      strings::StrAppend(out, "  for (int64 i = ", arg_range, ".first; i < ",
                         arg_range, ".second; ++i)\n");
      strings::StrAppend(out, "    this->", output_names[i],
                         ".push_back(Output(ret, i));\n");
    } else {
      strings::StrAppend(out, "  this->", output_names[i], " = Output(ret, ",
                         arg_range, ".first);\n");
    }
  }
}

string OpInfo::GetConstructorBody() const {
  const string scope_str = arg_names[0];

  string body;
  string return_on_error =
      strings::StrCat("if (!", scope_str, ".ok()) return;");

  strings::StrAppend(&body, "  ", return_on_error, "\n");

  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const auto& arg(op_def.input_arg(i));
    strings::StrAppend(&body, "  auto _", arg.name(), " = ::tensorflow::ops::",
                       ArgIsList(arg) ? "AsNodeOutList" : "AsNodeOut", "(",
                       scope_str, ", ", AvoidCPPKeywords(arg.name()), ");\n");
    strings::StrAppend(&body, "  ", return_on_error, "\n");
  }

  strings::StrAppend(&body, "  ::tensorflow::Node* ret;\n");
  strings::StrAppend(&body, "  const auto unique_name = ", scope_str,
                     ".GetUniqueNameForOp(\"", op_name, "\");\n");
  strings::StrAppend(
      &body, "  auto builder = ::tensorflow::NodeBuilder(unique_name, \"",
      graph_op_def.name(), "\")\n");
  const string spaces = "                     ";
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const auto& arg(op_def.input_arg(i));
    strings::StrAppend(&body, spaces, ".Input(_", arg.name(), ")\n");
  }
  for (int i = 0; i < op_def.attr_size(); ++i) {
    const auto& graph_attr(graph_op_def.attr(i));
    const auto& attr(op_def.attr(i));
    if (inferred_input_attrs.find(attr.name()) != inferred_input_attrs.end()) {
      continue;
    }
    const string attr_name = attr.has_default_value()
                                 ? strings::StrCat("attrs.", attr.name(), "_")
                                 : AvoidCPPKeywords(attr.name());
    strings::StrAppend(&body, spaces, ".Attr(\"", graph_attr.name(), "\", ",
                       attr_name, ")\n");
  }
  strings::StrAppend(&body, "  ;\n");
  strings::StrAppend(&body, "  ", scope_str, ".UpdateBuilder(&builder);\n");
  strings::StrAppend(&body, "  ", scope_str, ".UpdateStatus(builder.Finalize(",
                     scope_str, ".graph(), &ret));\n");
  strings::StrAppend(&body, "  ", return_on_error, "\n");

  // TODO(b/28152992): Enable this code-path once we have converted
  // all python shape functions to call their C++ versions.

  // strings::StrAppend(&body, "  ", scope_str, ".UpdateStatus(", scope_str,
  //                    ".refiner()->AddNode(ret));\n");

  GetOutput(&body);
  return body;
}

void OpInfo::WriteClassDef(WritableFile* cc) const {
  string class_def;
  strings::StrAppend(&class_def,
                     GetConstructorDecl(strings::StrCat(op_name, "::"),
                                        /* include_attr */ true),
                     " {\n");
  strings::StrAppend(&class_def, GetConstructorBody());
  strings::StrAppend(&class_def, "}\n\n");

  if (has_optional_attrs) {
    strings::StrAppend(&class_def,
                       GetConstructorDecl(strings::StrCat(op_name, "::"),
                                          /* include_attr */ false));
    strings::StrAppend(&class_def, "\n  : ", op_name, "(");
    int i = 0;
    for (; i < arg_names.size(); ++i) {
      if (i > 0) strings::StrAppend(&class_def, ", ");
      strings::StrAppend(&class_def, arg_names[i]);
    }
    if (i > 0) strings::StrAppend(&class_def, ", ");
    strings::StrAppend(&class_def, op_name, "::Attrs()");
    strings::StrAppend(&class_def, ") {}\n\n");
  }
  TF_CHECK_OK(cc->Append(class_def));
}

} // anon namespace


#define APPEND(x) TF_CHECK_OK(hx->Append( (x) ) )

string ToHaxeType(const string &val)
{
   if (val=="bool")
      return "Bool";
   else if (val=="float" || val=="double" || val=="float32" || val=="float64" )
      return "Float";
   else if (val=="int" || val=="int32" )
      return "Int";
   else if (val=="string")
      return "String";
   else if (val=="type" || val=="dtype")
      return "tf.Type";
   else if (val.substr(0,5)=="list(")
      return "Array<" + ToHaxeType(val.substr(5,val.size()-6)) + ">";
   else if (val=="")
      return "Dynamic";
   else if (val=="Index")
      return "Int";
   else if (val[0]=='T')
      return "tf.Tensor";
   else if (val=="shape") // TODO - 'either' ?
      return "Array<Int>";

   printf("  unknown haxe type %s - assuming type restriction\n", val.c_str());
   // Probably a type-restriction
   return "tf.Tensor";
}


string ToHaxeSuffix(const string &val)
{
   if (val.substr(0,5)=="list(")
      return ToHaxeSuffix(val.substr(5,val.size()-6)) + "Array";
   else if (val=="type" || val=="dtype")
      return "Type";
   else if (val[0]=='T')
      return "Tensor";
   else if (val=="shape") // TODO - 'either' ?
      return "Shape";
   string result = ToHaxeType(val);
   if (result=="tf.Tensor")
      return "Tensor";
   return result;
}

bool canDefaultType(const string &inType)
{
   string val = ToHaxeType(inType);
   return val=="Int" || val=="Float" || val=="Bool" || val=="String";
}

string StringToHaxe(const string& str) {
  return strings::StrCat("\"", str_util::CEscape(str), "\"");
}


string DataTypeToHaxe(DataType dtype, bool forArray = false) {
  switch (dtype) {
     case DT_FLOAT : return forArray ? "cpp.Float32" : "Float";
     case DT_DOUBLE : return "Float";
     case DT_INT32 : return "Int";
     case DT_UINT8 : return "Int";
     case DT_INT16 : return "Int";
     case DT_INT8 : return "Int";
     case DT_STRING : return "string";
     case DT_COMPLEX64 : return "tf.Complex";
     case DT_INT64 : return "haxe.Int64";
     case DT_BOOL : return "Bool";
     case DT_QINT8 : return "Int";
     case DT_QUINT8 : return "Int";
     case DT_QINT32 : return "Int";
     case DT_BFLOAT16 : return "Int";
     case DT_QINT16 : return "Int";
     case DT_QUINT16 : return "Int";
     case DT_UINT16 : return "Int";
     case DT_COMPLEX128 : return "tf.Complex";
     case DT_HALF : return "Float";
     case DT_RESOURCE : return "haxe.io.Bytes";
     default: ;
  }
  return "Dynamic";
}


string DataTypeToHaxeTfType(DataType dtype) {
  switch (dtype) {
     case DT_FLOAT : return "tf.Type.Float32";
     case DT_DOUBLE : return "tf.Type.Float64";
     case DT_INT32 : return "tf.Type.Int32";
     case DT_UINT8 : return "tf.Type.UInt32";
     case DT_INT16 : return "tf.Type.Int16";
     case DT_INT8 : return "tf.Type.Int8";
     case DT_STRING : return "ty.Type.StringType";
     case DT_COMPLEX64 : return "tf.Type.Complex64";
     case DT_INT64 : return "tf.Type.Int64";
     case DT_BOOL : return "tf.Type.BoolType";
     case DT_QINT8 : return "tf.Type.QInt8";
     case DT_QUINT8 : return "tf.Type.QUInt8";
     case DT_QINT32 : return "tf.Type.QInt32";
     case DT_BFLOAT16 : return "tf.Type.BFloat16";
     case DT_QINT16 : return "tf.Type.QInt16";
     case DT_QUINT16 : return "tf.Type.QUInt16";
     case DT_UINT16 : return "tf.Type.UInt16";
     case DT_COMPLEX128 : return "tf.Type.Complex128";
     case DT_HALF : return "tf.Type.Half";
     case DT_RESOURCE : return "tf.Type.Resource";
     default: ;
  }
  return "null";
}


string ShapeToHaxe(const TensorShapeProto& shape) {
  string haxe = "[";
  for (const auto& dim : shape.dim()) {
    if (haxe.size() > 1) strings::StrAppend(&haxe, ", ");
    if (!dim.name().empty()) {
      strings::StrAppend(&haxe, "(", StringToHaxe(dim.name()), "\", ",
                         dim.size(), ")");
    } else {
      strings::StrAppend(&haxe, dim.size());
    }
  }
  strings::StrAppend(&haxe, "]");
  return haxe;
}

string AttrListToHaxe(const AttrValue& value) {
  string ret;
  if (value.list().s_size() > 0) {
    for (int i = 0; i < value.list().s_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, StringToHaxe(value.list().s(i)));
    }
  } else if (value.list().i_size() > 0) {
    for (int i = 0; i < value.list().i_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, value.list().i(i));
    }
  } else if (value.list().f_size() > 0) {
    for (int i = 0; i < value.list().f_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, value.list().f(i));
    }
  } else if (value.list().b_size() > 0) {
    for (int i = 0; i < value.list().b_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, value.list().b(i) ? "true" : "false");
    }
  } else if (value.list().type_size() > 0) {
    for (int i = 0; i < value.list().type_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, DataTypeToHaxe(value.list().type(i)));
    }
  } else if (value.list().shape_size() > 0) {
    for (int i = 0; i < value.list().shape_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, ShapeToHaxe(value.list().shape(i)));
    }
  }
  return ret;
}

string getHaxeValue(const string& type, const AttrValue& value) {
  if (type == "string") {
    return "\"" + value.s() + "\"";
  } else if (type == "int") {
    return strings::StrCat(value.i());
  } else if (type == "float") {
    return strings::StrCat(value.f());
  } else if (type == "bool") {
    return value.b() ? "true" : "false";
  } else if (type == "type") {
    return DataTypeToHaxeTfType(value.type());
  } else if (type == "shape") {
    return ShapeToHaxe(value.shape());
  } else if (type == "tensor") {
    return "tf.Tensor";
  } else {
    return strings::StrCat("[", AttrListToHaxe(value), "]");
  }
}

string remapName(const string &inName)
{
   if (inName=="var")
      return "variable";
   if (inName=="cast")
      return "typeCast";
   if (inName=="switch")
      return "switchVal";
   return inName;
}

string toHaxeCase(const string inValue)
{
   char chars[2];
   chars[0] = inValue[0];
   if (chars[0]>='A' && chars[0]<='Z')
   {
      chars[0] += 'a' - 'A';
      chars[1] = 0;
      return chars + inValue.substr(1, inValue.size()-1);
   }
   return inValue;
}


void genHxcppCode(HxString className, HxString classFile, HxString inFilter)
{
   int argc = 1;
   char app[] = "app";
   char* argv[] = {app};
   tensorflow::port::InitMain(app, &argc, (char ***)&argv);

   OpList ops;
   OpRegistry::Global()->Export(false, &ops);
   std::string overrides = "../modules/tensorflow/tensorflow/cc/ops/op_gen_overrides.pbtxt";

   std::string filter(inFilter.c_str());

   Env* env = Env::Default();

   // Load the override map.
   OpGenOverrideMap override_map;
   if (!overrides.empty())
   {
     TF_CHECK_OK(override_map.LoadFileList(env, overrides));
   }

   std::unique_ptr<WritableFile> hx = nullptr;

   for (const auto& graph_op_def : ops.op())
   {
      // Skip deprecated ops.
      if (graph_op_def.has_deprecation() &&
          graph_op_def.deprecation().version() <= TF_GRAPH_DEF_VERSION)
      {
        continue;
      }

      string name = graph_op_def.name();
      // Filter by category...
      if (filter.find(":"+name+":")==string::npos)
         continue;

      // Incorporate overrides from override_map.
      OpDef interface_op_def = graph_op_def;
      const OpGenOverride* op_override =
          override_map.ApplyOverride(&interface_op_def);
      std::vector<string> aliases;
      if (op_override)
      {
        if (op_override->skip())
           continue;
        aliases.assign(op_override->alias().begin(), op_override->alias().end());
        if (op_override->hide())
        {
          continue;
        }
      }

      // This isn't a hidden op, write it to the main files.
      OpInfo op_info(graph_op_def, interface_op_def, aliases);
      if (!hx)
      {
         TF_CHECK_OK(env->NewWritableFile(classFile.c_str(), &hx));
         APPEND("// AutoGen APIn\n");
         APPEND("package tf;\n\n");
         APPEND(string("class ") + className.c_str() + "{\n");
      }

      APPEND("public static function " + remapName(toHaxeCase(op_info.op_name)) + "(?inNodeName:String\n");

      const OpDef &op_def = op_info.op_def;
      std::string indent = "\t\t\t,";
      std::string bodyI = "\t";
      std::string defaultArgs = "";
      std::string addInputs = "";
      std::string addAttrs = "";
      std::string returnOutputs = "";


      for (int i = 0; i < op_def.input_arg_size(); ++i)
      {
         const auto& arg(op_def.input_arg(i));

         std::string inputSuffix = "";
         string name = remapName(arg.name());
         if (!arg.type_attr().empty())
             APPEND( indent + name + ": tf.Output\n");
         else if (ArgIsList(arg))
         {
             inputSuffix = "Array";
             APPEND( indent + name + ": Array< tf.Output >\n");
         }
         else
             APPEND( indent + name + ": tf.Output\n");

         addInputs += bodyI + "ctx.addInput" + inputSuffix +  "(" + name + ");\n";
      }

      for (int i = 0; i < op_def.attr_size(); ++i)
      {
         const auto& attr(op_def.attr(i));
         // Defines type restriction
         if (attr.name().substr(0,1)=="T")
            continue;

         string name = remapName(attr.name());
         if (  attr.has_default_value() )
         {
            if (canDefaultType(attr.type()))
            {
               APPEND( indent + name + ":" + ToHaxeType(attr.type()) );
               APPEND(" = " + getHaxeValue(attr.type(),attr.default_value()) );
            }
            else
            {
               APPEND( indent + "?" + name + ":" + ToHaxeType(attr.type()) );
               string defValue = getHaxeValue(attr.type(),attr.default_value());
               if (defValue=="tf.Tensor")
                  defValue = "ctx.get_" + name + "()";
               defaultArgs += "\tif (" + name + "==null) " + name + " = " + defValue + ";\n";
            }
         }
         else
            APPEND( indent + name + ":" + ToHaxeType(attr.type()) );

         APPEND("\n");

         std::string attribSuffix = ToHaxeSuffix(attr.type());
         addAttrs += bodyI + "ctx.addAttrib" + attribSuffix +  "(\"" + attr.name()+ "\"," +name + ");\n";
      }


      std::vector<string> output_types;
      bool same = true;
      bool isArray = false;
      // Process outputs
      for (int i = 0; i < op_def.output_arg_size(); ++i) {
        const auto& arg = op_def.output_arg(i);
        bool is_list = ArgIsList(arg);
        isArray = is_list;
        output_types.push_back( is_list ? "Array<tf.Output>" : "tf.Output");
        for(int j=0;j<output_types.size()-1;j++)
           if (output_types[j]!=output_types[output_types.size()-1])
              same = false;
      }

      if (output_types.size()==0)
      {
         APPEND("\t\t): Void {\n");
      }
      else if (output_types.size()==1)
      {
         APPEND("\t\t): " + output_types[0] + " {\n");
         returnOutputs = bodyI + (isArray ? "return ctx.endForOutputArray();\n" : "return ctx.endForOutput();\n");
      }
      else if (same)
      {
         APPEND("\t\t): Array<" + output_types[0] + "> {\n");
         returnOutputs = bodyI + "return ctx.endForOutputArray();\n";
      }
      else
      {
         APPEND("\t\t): Array<Dynamic> {\n");
         returnOutputs = bodyI + "return ctx.endForDynamicArray();\n";
      }

      APPEND(bodyI + string("var ctx = tf.Context.current;\n"));
      APPEND(defaultArgs);
      APPEND(bodyI + string("ctx.beginOp(\"") + name + "\",inNodeName);\n");
      APPEND(addInputs);
      APPEND(addAttrs);
      APPEND(returnOutputs);
      APPEND("}\n\n");
      }

   if (hx)
   {
      TF_CHECK_OK(hx->Append("}\n"));
      TF_CHECK_OK(hx->Close());
   }
}

DEFINE_PRIME3v(genHxcppCode);

void genCppCode(HxString cFilename, HxString hFilename)
{
   int argc = 1;
   char app[] = "app";
   char* argv[] = {app};
   tensorflow::port::InitMain(app, &argc, (char ***)&argv);

   OpList ops;
   OpRegistry::Global()->Export(false, &ops);
   std::string overrides = "../modules/tensorflow/tensorflow/cc/ops/op_gen_overrides.pbtxt";
   WriteCCOps(ops, hFilename.c_str(), cFilename.c_str(), overrides);
}

DEFINE_PRIME2v(genCppCode);


}  // namespace tensorflow

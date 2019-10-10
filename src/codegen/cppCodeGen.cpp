// Copyright 2018 The AITS DNNC Authors.All Rights Reserved.
//
// Licensed to the Apache Software Foundation(ASF) under one
// or more contributor license agreements.See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.See the License for the
// specific language governing permissionsand limitations
// under the License.
//
// This file is part of AITS DNN compiler maintained at
// https://github.com/ai-techsystems/dnnCompiler
//

#include <assert.h>
#include <codegen/cppCodeGen.h>

bool dnnc::cppCodeGen::write() {

  _includes.clear();

  std::string code = "";

  for (dnnParameters param : _graph.parameters()) {
    code += write(param);
  }
  for (ioNode *term : _graph.inputs()) {
    code += write(*term);
  }

  for (node *n : _graph) {
    if (n->ntype() == node::OPERATOR)
      code += write(*dynamic_cast<opNode *>(n));
  }

  for (ioNode *term : _graph.outputs()) {
    code += write(*term);
  }

  
  std::cout << writeIncludes() << "\n";
  std::cout << writeMainFunction(code) << "\n";
  return code.length();
}

std::string dnnc::cppCodeGen::writeIncludes() {
  std::string code ;
  for(auto& s : _includes)
    code += std::string("#include \"") + s + "\"\n";
  return code;
}

std::string dnnc::cppCodeGen::writeMainFunction(std::string body) {
  std::string code = "int main() {\n";
  code += body + "\n";
  code += _tab + "return 0;\n";
  code += "}\n";
  return code;
}


std::pair<std::string, std::string>
dnnc::cppCodeGen::initializeData(irTypeData dtype) {
  std::string varType;  // int, float, std::vector<float> etc
  std::string initData; // = {1.3, 1.5} etc
  switch (dtype.type()) {
  case IR_DataType::INT8:
  case IR_DataType::INT16:
  case IR_DataType::INT32:
  case IR_DataType::INT64: {
    std::vector<int> values = std::vector<int>(dtype);
    for (auto el : values)
      initData += (initData.size() ? "," : "{") + std::to_string(el);
    initData += values.size() ? "}" : "";
    varType = getDNNC_IRTypeStr(dtype.type());
    varType = values.size() ? "std::vector<" + varType + ">" : varType + "\n";
    break;
  }
  case IR_DataType::UINT8:
  case IR_DataType::UINT16:
  case IR_DataType::UINT32:
  case IR_DataType::UINT64: {
    std::vector<unsigned int> values = std::vector<unsigned int>(dtype);
    for (auto el : values)
      initData += (initData.size() ? "," : "{") + std::to_string(el);
    initData += values.size() ? "}" : "";
    varType = getDNNC_IRTypeStr(dtype.type());
    varType = values.size() ? "std::vector<" + varType + ">" : varType + "\n";
    break;
  }
  case IR_DataType::FLOAT:
  case IR_DataType::FLOAT16:
  case IR_DataType::DOUBLE: {
    std::vector<float> values = std::vector<float>(dtype);
    for (auto el : values)
      initData += (initData.size() ? "," : "{") + std::to_string(el);
    initData += values.size() ? "}" : "";
    varType = getDNNC_IRTypeStr(dtype.type());
    varType = values.size() ? "std::vector<" + varType + ">" : varType + "\n";
    break;
  }
  case IR_DataType::STRING:
    varType = "std::string";
    initData = std::string(dtype);
    break;
  case IR_DataType::TENSOR_BOOL:
    // TODO:
    break;
  case IR_DataType::TENSOR_INT:
    // TODO:
    break;
  case IR_DataType::TENSOR_FLOAT:
    // TODO:
    break;
  default:
    assert(false && "irTypeData object created without type");
    break;
  }
  return std::pair<std::string, std::string>(varType, initData);
}

std::string dnnc::cppCodeGen::write(dnnParameters param) {
  std::pair<std::string, std::string> var = initializeData(param.data());
  return var.first + " " + param.name() + " = " + var.second + " ;\n";
}

std::string dnnc::cppCodeGen::write(ioNode &term) {
  std::string code = "";
  return code;
}

std::string dnnc::cppCodeGen::write(node &n) {

  std::string code;

  assert(n.ntype() == node::OPERATOR);

  opNode computeNode = dynamic_cast<opNode &>(n);

  assert(computeNode.symbol() != opInvalid);

  std::string opCode = getOpCodeStr(computeNode.symbol());
  _includes.push_back("operators/" + opCode + ".h");

  std::string node_name = computeNode.name();

  assert(node_name.length());

  // binary operators
  std::vector<std::string> ins = computeNode.inputs();
  std::vector<std::string> outs = computeNode.outputs();
  if (ins.size() != 2 || outs.size() != 1)
    return code;

  assert(ins.size() == 2 && outs.size() == 1);

  std::string outType = "float"; // TODO: levelize graph, infer out types.
                                 // getDNNC_IRTypeStr(outNode.type());
  std::string in1Type = "float", in2Type = "float";

  // Step 1: Instantiate opterator
  code +=
      _tab + opCode + "<" + outType + "> " + node_name + "(\"" + node_name + "\");\n";

  // Step 2: Add attribute
  for (nodeAttribute attr : computeNode) {
    std::string name = getAttrNameStr(attr.name());
    std::pair<std::string, std::string> var = initializeData(attr.data());
    code += _tab + var.first + " " + name + " = " + var.second + " ;\n";
    code += _tab + node_name + ".setAttribute ( attr_" + name + ", " + name + " );\n";
  }

  // Step 3: Add compute function.
  code += _tab + "tensor<" + outType + "> " + node_name + "_" + outs[0] + " = " +
          node_name + ".compute ( " + node_name + "_" + ins[0] + ", " +
          node_name + "_" + ins[1] + ");\n";

  return code + "\n";
}

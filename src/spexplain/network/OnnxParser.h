//
// Created by labbaf on 26.12.2025.
//

#ifndef XAI_SMT_ONNXPARSER_H
#define XAI_SMT_ONNXPARSER_H

#include <spexplain/common/List.h>
#include <spexplain/common/Map.h>
#include <spexplain/common/MString.h>
#include <spexplain/common/Set.h>
#include <spexplain/common/TensorUtils.h>
#include <spexplain/common/Vector.h>
#include <onnx-1.15.0/onnx.pb.h>

#include <memory>

#define ONNX_LOG( x, ... ) LOG( GlobalConfiguration::ONNX_PARSER_LOGGING, "OnnxParser: %s\n", x )

namespace spexplain {
class Network2;

class OnnxParser
{
public:
    static void parse(
                       const std::string_view path,
                       const Set<String> inputNames,
                       const Set<String> outputNames );

    static std::unique_ptr<Network2> buildNetwork2(std::string_view path);


private:
    // Settings //
    OnnxParser(
                const std::string_view path,
                const Set<String> inputNames,
                const Set<String> terminalNames,
                Network2 *net = nullptr );

    // InputQueryBuilder &_query;
    onnx::GraphProto _network;
    Set<String> _inputNames;

    /// Non-owning pointer; non-null only when building a Network2 object.
    Network2 *_net;

    /*
      The set of terminal nodes for the query. Note that these doesn't have to be outputs of
      the network, they can be intermediate nodes.
    */
    Set<String> _terminalNames;

    // State //

    Map<String, TensorShape> _shapeMap;
    // Map<String, Vector<Variable>> _varMap;
    Map<String, const Vector<int64_t>> _constantIntTensors;
    Map<String, const Vector<double>> _constantFloatTensors;
    Map<String, const Vector<int32_t>> _constantInt32Tensors;
    Set<String> _processedNodes;
    unsigned _numberOfFoundInputs;

    // Methods //

    const Set<String> readInputNames();
    const Set<String> readOutputNames();
    void validateUserInputNames( const Set<String> &inputNames );
    void validateUserTerminalNames( const Set<String> &terminalNames );

    void readNetwork( const String &path );
    void initializeShapeAndConstantMaps();
    void validateAllInputsAndOutputsFound();

    void processGraph();
    void processNode( String &nodeName, bool makeEquations );
    void makeNodeObjects( onnx::NodeProto &node, bool makeEquations ); //TODO
    Set<String> getInputsToNode( onnx::NodeProto &node );
    List<onnx::NodeProto> getNodesWithOutput( String &nodeName );
    // Vector<Variable> makeNodeVariables( String &nodeName, bool isInput );

    bool isConstantNode( String name );

    void transferValues( String oldName, String newName );
    void insertConstant( String name, const onnx::TensorProto &tensor, TensorShape shape );

    void constant( onnx::NodeProto &node );
    void identity( onnx::NodeProto &node );
    void dropout( onnx::NodeProto &node );
    void cast( onnx::NodeProto &node );
    void reshape( onnx::NodeProto &node );
    void squeeze( onnx::NodeProto &node );
    void unsqueeze( onnx::NodeProto &node );
    void flatten( onnx::NodeProto &node );
    void transpose( onnx::NodeProto &node );
    // void batchNormEquations( onnx::NodeProto &node, bool makeEquations );
    void maxPoolEquations( onnx::NodeProto &node, bool makeEquations );
    void convEquations( onnx::NodeProto &node, bool makeEquations );
    void gemmEquations( onnx::NodeProto &node, bool makeEquations );
    // void scaleAndAddEquations( onnx::NodeProto &node,
    //                            bool makeEquations,
    //                            double coefficient1,
    //                            double coefficient2 );
    void matMulEquations( onnx::NodeProto &node, bool makeEquations );
    void reluEquations( onnx::NodeProto &node, bool makeEquations );
    void sigmoidEquations( onnx::NodeProto &node, bool makeEquations );
};
}

#endif // XAI_SMT_ONNXPARSER_H

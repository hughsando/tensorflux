import tf.TrainingOps;
import tf.StringOps;
import tf.StateOps;
import tf.SparseOps;
import tf.RandomOps;
import tf.ParsingOps;
import tf.NnOps;
import tf.MathOps;
import tf.LoggingOps;
import tf.LinalgOps;
import tf.IoOps;
import tf.ImageOps;
import tf.FunctionalOps;
import tf.DataFlowOps;
import tf.CtcOps;
import tf.ControlFlowOps;
import tf.CandidateSamplingOps;
import tf.ArrayOps;
import tf.ConstOps;
import tf.Tensor;
import tf.Const;

class Test
{
   #if cpp
   public static function __init__()
   {
      cpp.Lib.pushDllSearchPath( "../../ndll/" + cpp.Lib.getBinDirectory() );
   }
   #end

   public static function main()
   {
      // Basic constant operations
      // The value returned by the constructor represents the output
      // of the Constant op.
      var a = Const.int32(2);
      var b = Const.int32(3);
      var sum = MathOps.add(a,b);

      //var hello = Tensor.fromString(value, type);
/*
      var session = tf.Session.get();
      Sys.println("a=2, b=3")
      Sys.println("Addition with constants: " + sess.run(a+b))
      Sys.println("Multiplication with constants: " + sess.run(a*b))

      // Basic Operations with variable as graph input
      // The value returned by the constructor represents the output
      // of the Variable op. (define as input when running session)
      // tf Graph input
      a = ArrayOps.Placeholder(tf.Type.Int16)
      b = ArrayOps.Placeholder(tf.Type.Int16)

      // Define some operations
      add = tf.add(a, b)
      mul = tf.muliply(a, b)

      // Launch the default graph.
      with tf.Session() as sess:
          // Run every operation with variable input
          print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
          print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))


// ----------------
// More in details:
// Matrix Multiplication from TensorFlow official tutorial

// Create a Constant op that produces a 1x2 matrix.  The op is
// added as a node to the default graph.
//
// The value returned by the constructor represents the output
// of the Constant op.
matrix1 = tf.constant([[3., 3.]])

// Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

// Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
// The returned value, 'product', represents the result of the matrix
// multiplication.
product = tf.matmul(matrix1, matrix2)

// To run the matmul op we call the session 'run()' method, passing 'product'
// which represents the output of the matmul op.  This indicates to the call
// that we want to get the output of the matmul op back.
//
// All inputs needed by the op are run automatically by the session.  They
// typically are run in parallel.
//
// The call 'run(product)' thus causes the execution of threes ops in the
// graph: the two constants and matmul.
//
// The output of the op is returned in 'result' as a numpy `ndarray` object.
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
    // ==> [[ 12.]]
*/

   }
}

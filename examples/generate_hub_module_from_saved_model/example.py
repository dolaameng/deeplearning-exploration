import tensorflow as tf
import tensorflow_hub as hub
import os

## create a simple saved_model
def generate_saved_model(export_dir):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[], name='x')
        # pretend there is some training here
        w = tf.get_variable(name='w', shape=[], dtype=tf.float32)
        train_op = tf.assign(w, 3.14)
        y = tf.multiply(x, w, name='y')
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(train_op)
            print(sess.run(y, feed_dict={x: 10}))
            
            # save it to a saved_model, with 'serve' tag
            tf.saved_model.simple_save(sess, 
                                       export_dir=export_dir, 
                                       inputs={'x': x}, 
                                       outputs={'y':y})
    return export_dir


## load a saved_model with variables, convert it to a frozen model
def generate_frozen_graph_from_saved_model(saved_model_dir):
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            meta_graph_def = tf.saved_model.loader.load(sess, ['serve'], saved_model_dir)
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, 
                                                                            graph.as_graph_def(), 
                                                                            output_node_names=['y'])
            logdir = saved_model_dir
            name = 'frozen_graph.pb'
            tf.train.write_graph(frozen_graph_def, 
                                 logdir=logdir, 
                                 name=name, 
                                 as_text=False)
    return os.path.join(logdir, name)


## convert the frozen graph into a hub module
## you don't have to worry about the variable copying in this case
## because everything is in constants. On the other side, the model
## cannot be further fine-tunned either.

def get_module_fn(frozen_graph_file):
    def module_fn():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        hub.add_signature('default', inputs={'x': x}, outputs={'y': y})
    return module_fn

def export_hub_from_frozen_graph(frozen_graph_file, hub_dir):
    spec = hub.create_module_spec(module_fn=get_module_fn(frozen_graph_file))
    with tf.Graph().as_default():
        m = hub.Module(spec)
        with tf.Session() as sess:
            m.export(hub_dir, sess)


def test_hub_module(hub_dir):
    ## test generated hub
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[])
        m = hub.Module(hub_dir)
        r = m(x, as_dict=True)
        with tf.Session() as sess:
            print(sess.run(r, feed_dict={x: 10}))


if __name__ == '__main__':
    export_dir = '/tmp/simple_model'
    saved_model_dir = generate_saved_model(export_dir)
    frozen_graph_file = generate_frozen_graph_from_saved_model(saved_model_dir)
    hub_dir = '/tmp/simple_model_hub'
    export_hub_from_frozen_graph(frozen_graph_file, hub_dir)
    test_hub_module(hub_dir)
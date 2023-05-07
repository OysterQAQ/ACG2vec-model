from absl import app, flags
from clip_tf import converter


FLAGS = flags.FLAGS
flags.DEFINE_enum('model', 'RN50', converter.MODELS.keys(), 'CLIP model architecture to convert')
flags.DEFINE_string('output', 'models/CLIP_{model}', 'CLIP Keras SavedModel Output destination')
flags.DEFINE_string('output_path', 'models/CLIP_{model}', 'CLIP Keras SavedModel Output destination')
flags.DEFINE_string('image_output', None, 'Image encoder Keras SavedModel output destination (optional)')
flags.DEFINE_string('text_output', None, 'Text encoder Keras SavedModel output destination (optional)')
flags.DEFINE_string('model_path', None, '')
flags.DEFINE_bool('img_base64', None, '')
flags.DEFINE_bool('all', False, 'Export all versions. (will use output location if image_output or text_output are not present)')


def main(argv):
    converter.convert(FLAGS.model, FLAGS.output, FLAGS.image_output, FLAGS.text_output, FLAGS.all,model_path=FLAGS.model_path,full_output_path=FLAGS.output_path+'_full',img_output_path=FLAGS.output_path+'_img',text_output_path=FLAGS.output_path+'_text'
                      ,img_base64=FLAGS.img_base64)


if __name__ == '__main__':
    app.run(main)

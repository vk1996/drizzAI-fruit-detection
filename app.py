from yolox_onnx import YOLOX_ONNX
import gradio as gr
import cv2
import numpy as np

def show_example(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def get_response(input_img,confidence_threshold,iou_threshold):

    '''
    detects all possible pedestrians in the image and recognizes it
        Args:
            input_img (numpy array): one image of type numpy array
            confidence_threshold (float) : minimum confidence prob required for bounding box candidate
            iou_threshold (float)  : intersection threshold above which bounding box will be neglected

        Returns:
            return img(numpy array): image with bounding boxes of pedestrians
    '''


    if not hasattr(input_img,'shape'):
        return "invalid input",input_img

    input_img=cv2.cvtColor(input_img,cv2.COLOR_RGB2BGR).astype(np.float32)

    detector.predict(input_img,confidence_threshold,iou_threshold)
    out_img=detector.output_img.astype(np.uint8)

    return cv2.cvtColor(out_img,cv2.COLOR_BGR2RGB).astype(np.uint8)


if __name__ == "__main__":


    detector=YOLOX_ONNX('models/fruit-detection-epoch120.onnx')
    iface = gr.Interface(
        cache_examples=False,
        fn=get_response,
        inputs=[gr.Image(type="numpy"),  # Accepts image input
        gr.Slider(0, 1,value=0.4, step=0.01, label="confidence_threshold"),
        gr.Slider(0, 1,value=0.4, step=0.01, label="iou_threshold")],
        examples=[[show_example('test-images/test2.png')],[show_example('test-images/test3.jpeg')],[show_example('test-images/test1.jpeg')]],
        outputs=[gr.Image(type="numpy")],
        title="Drizz AI fruit detection",
        description="Upload images for fruit (apple,banana,grapes,orange,strawberry) detection")

    iface.launch()

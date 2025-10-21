package com.example.flowerclassifierapp

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.flowerclassifierapp.ui.theme.FlowerClassifierAppTheme
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : ComponentActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Load TFLite model and labels from assets
        val tfliteModel = FileUtil.loadMappedFile(this, "model.tflite")
        interpreter = Interpreter(tfliteModel)
        labels = FileUtil.loadLabels(this, "flower_labels.txt")

        setContent {
            FlowerClassifierAppTheme {
                FlowerClassifierScreen(interpreter, labels)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FlowerClassifierScreen(
    interpreter: Interpreter,
    labels: List<String>
) {
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var resultText by remember { mutableStateOf("Select an image to classify") }

    val context = LocalContext.current

    val imagePickerLauncher =
        rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                try {
                    // Decode bitmap safely to avoid hardware bitmap issues
                    val source = ImageDecoder.createSource(context.contentResolver, it)
                    val tmpBitmap = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                        decoder.isMutableRequired = true
                    }
                    val selectedBitmap = tmpBitmap.copy(Bitmap.Config.ARGB_8888, true)
                    bitmap = selectedBitmap

                    // Classify the image
                    resultText = classifyImage(selectedBitmap, interpreter, labels)
                } catch (e: Exception) {
                    resultText = "Error processing image: ${e.message}"
                }
            }
        }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Flower Classifier") })
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            bitmap?.let {
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier
                        .size(250.dp)
                        .padding(8.dp)
                )
            }

            Text(
                text = resultText,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(top = 16.dp)
            )

            Spacer(modifier = Modifier.height(16.dp))

            Button(onClick = { imagePickerLauncher.launch("image/*") }) {
                Text("Select Image")
            }
        }
    }
}

fun classifyImage(
    bitmap: Bitmap,
    interpreter: Interpreter,
    labels: List<String>,
    normalizeTo: Pair<Float, Float> = 0f to 1f // default normalization [0,1]
): String {
    // Get model input shape dynamically
    val inputShape = interpreter.getInputTensor(0).shape() // [1, height, width, 3]
    val inputHeight = inputShape[1]
    val inputWidth = inputShape[2]

    // Resize bitmap to match model input
    val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)

    // Allocate buffer
    val inputBuffer = ByteBuffer.allocateDirect(4 * inputWidth * inputHeight * 3)
    inputBuffer.order(ByteOrder.nativeOrder())

    // Fill buffer with normalized pixel data
    val pixels = IntArray(inputWidth * inputHeight)
    resizedBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

    val (minVal, maxVal) = normalizeTo
    val scale = maxVal - minVal

    for (pixel in pixels) {
        // Extract RGB channels and normalize like in Python
        val r = ((pixel shr 16) and 0xFF) / 255.0f * scale + minVal
        val g = ((pixel shr 8) and 0xFF) / 255.0f * scale + minVal
        val b = (pixel and 0xFF) / 255.0f * scale + minVal

        inputBuffer.putFloat(r)
        inputBuffer.putFloat(g)
        inputBuffer.putFloat(b)
    }
    inputBuffer.rewind()

    // Prepare output buffer
    val outputBuffer = TensorBuffer.createFixedSize(
        intArrayOf(1, labels.size),
        DataType.FLOAT32
    )

    // Run inference
    interpreter.run(inputBuffer, outputBuffer.buffer.rewind())

    // Get top prediction
    val probabilities = outputBuffer.floatArray
    val topIdx = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
    val confidence = probabilities[topIdx] * 100

    return "${labels[topIdx]} (%.2f%%)".format(confidence)
}



package com.example.density.service;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Service
public class ImageService {

    static {
        try {
            System.out.println("OpenCV 라이브러리 로드 시도");
            System.load("C:\\Users\\A\\Desktop\\density\\libs\\opencv_java452.dll");
            System.out.println("OpenCV 라이브러리 로드 성공");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("OpenCV 라이브러리 로드 실패");
            e.printStackTrace();
        }
    }

    private static final String MODEL_CONFIGURATION = "C:\\Users\\A\\Desktop\\density\\yolov3.cfg";
    private static final String MODEL_WEIGHTS = "C:\\Users\\A\\Desktop\\density\\yolov3.weights";
    private static final List<String> LAYER_NAMES = Arrays.asList("yolo_82", "yolo_94", "yolo_106");

    private Net net;

    public ImageService() {
        try {
            System.out.println("모델 구성 파일 경로: " + MODEL_CONFIGURATION);
            System.out.println("모델 가중치 파일 경로: " + MODEL_WEIGHTS);
            System.out.println("OpenCV 네트워크 초기화 시도");
            net = Dnn.readNetFromDarknet(MODEL_CONFIGURATION, MODEL_WEIGHTS);
            System.out.println("OpenCV 네트워크 초기화 성공");
        } catch (Exception e) {
            System.err.println("OpenCV 네트워크 초기화 실패");
            e.printStackTrace();
        }
    }

    public String processImage(MultipartFile file) {
        File tempFile = new File(System.getProperty("java.io.tmpdir") + System.currentTimeMillis() + file.getOriginalFilename());
        try {
            file.transferTo(tempFile);
        } catch (IOException e) {
            e.printStackTrace();
            return "파일 업로드 실패";
        }

        Mat image = Imgcodecs.imread(tempFile.getAbsolutePath());
        if (image.empty()) {
            return "이미지 로드 실패";
        }

        List<Mat> outputs = new ArrayList<>();
        Mat blob = Dnn.blobFromImage(image, 1 / 255.0, new Size(416, 416), new Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        net.forward(outputs, LAYER_NAMES);

        int personCount = 0;
        float confidenceThreshold = 0.5f;

        for (Mat result : outputs) {
            for (int i = 0; i < result.rows(); i++) {
                Mat row = result.row(i);
                Mat scores = row.colRange(5, result.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);

                if (mm.maxVal > confidenceThreshold) {
                    int classId = (int) mm.maxLoc.x;
                    if (classId == 0) { // Class ID 0 corresponds to "person" in COCO dataset
                        personCount++;
                    }
                }
            }
        }

        tempFile.delete();

        return "밀집도 계산 결과: " + personCount + "명";
    }
}

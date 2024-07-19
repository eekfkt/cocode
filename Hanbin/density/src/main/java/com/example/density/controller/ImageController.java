package com.example.density.controller;

import com.example.density.service.ImageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

@Controller
public class ImageController {

    @Autowired
    private ImageService imageService;

    @GetMapping("/")
    public String index() {
        return "index";
    }

    @PostMapping("/upload")
    public String uploadImage(@RequestParam("image") MultipartFile file, Model model) {
        String densityResult = imageService.processImage(file);
        model.addAttribute("density", densityResult);
        return "uploadStatus";
    }
}

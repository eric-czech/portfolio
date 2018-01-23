package com.test.deconv;

import java.util.LinkedHashMap;
import java.util.Map;

import deconvolution.algorithm.RichardsonLucy;
import deconvolutionlab.Lab;
import signal.RealSignal;
import signal.factory.Gaussian;

public class Test {

	static int deconvolve(String file) {
		RealSignal image  = Lab.openFile(file);
		
		double sigma = 5.;
		RealSignal psf = new Gaussian(sigma, sigma, sigma)
				.generate(image.nx, image.ny, image.nz);
		
		RichardsonLucy algo = new RichardsonLucy(10);
		algo.run(image, psf);
		int iter = algo.getIterations();
		return iter;
	}
	
	public static Map<String, Long> profile() {
		String path = "/Users/eczech/data/research/hammer/akoya/CODEX/CODEX_Runs/20180109/Cycle1_";
		
		Map<String, Long> times = new LinkedHashMap<String, Long>();
		for (int i = 1; i <= 10; i++) {
			long start = System.currentTimeMillis();
			String file = String.format("Image_000%02d_Z001_Overlay.tif", i);
			file = path + "/" + file;
			deconvolve(file);
			long time = (long)Math.floor((System.currentTimeMillis() - start) / 1000);
			times.put(file, time);
		}
		return times;
	}
	
	static void example() {
		String path = "/Users/eczech/data/research/hammer/akoya/CODEX/CODEX_Runs/20180109/Cycle1_";
		String file = "Image_00004_Z002_Overlay.tif";
		RealSignal image  = Lab.openFile(path + "/" + file);
		
		double sigma = 5.;
		RealSignal psf = new Gaussian(sigma, sigma, sigma)
				.generate(image.nx, image.ny, image.nz);
		
		RealSignal r = new RichardsonLucy(10).run(image, psf);
		Lab.show(image, "Original");
		Lab.show(psf, "PSF");
//		Lab.show(r, "Result");
	}
	
	public static void main(String[] args) {
		
		Map<String, Long> times = profile();
		System.out.println(times.values());
//		
//		deconvolve("/Users/eczech/data/research/hammer/akoya/"
//				+ "CODEX/CODEX_Runs/20180109/Cycle1_/"
//				+ "Image_00004_Z002_Overlay.tif");
	}
}

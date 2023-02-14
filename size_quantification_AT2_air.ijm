function measureSize(input, output, filename) {

open(input + "\" + filename);   
run("Set Scale...", "distance=461.0011 known=1000 unit=µm global");

run("8-bit");

setAutoThreshold("Default");
run("Threshold...");
setThreshold(5, 255);
setOption("BlackBackground", false);
run("Convert to Mask");
run("Fill Holes");
run("Analyze Particles...", "size=100.00-Infinity display clear add");
resultsfname = replace(filename, '.tif', '.csv');
saveAs("Results", output + '/' + resultsfname);

selectWindow(filename);
saveAs("Tiff", output + '/' + filename);
close();
}

run("Close All");

input = "C:\Users\44749\Documents\PhD work\day7_nonmut_gf_to_scan";
output = "C:\Users\44749\Documents\PhD work\day7_nonmut_gf_scanned";
format = ".tif"


filelist = getFileList(input); // obtain a list of files in a directory
for (i = 0; i < filelist.length; i++){
	if (endsWith(filelist[i], format) == true){
        measureSize(input, output, filelist[i]);
		print(filelist[i]);
        imagename = File.getName(filelist[i]);
        print("Processing: ", imagename);
	}
}
print("Done.");

// run background subtraction
//run("Subtract Background...", "rolling=1000 sliding");

//run thresholding
//setAutoThreshold("Default"); //automatic
//run("Enhance Contrast", "saturated=0.35");
//setMinAndMax(1, 46);
//setThreshold(9, 255); // defined threshold
//setOption("BlackBackground", false);
//run("Convert to Mask");

//run erode/dilate/fill hole/watershed for BSC colonies
//run("Open");
//run("Fill Holes");
//run("Erode");
//run("Erode");
//run("Dilate");
//run("Dilate");
//run("Watershed");


<?php
// modules/read_file.php, Jake Deery, 2018
if(!function_exists("array_1d_to_2d")) {
	include "array_1d_to_2d.php";
}

// read a file and return an array
function read_file($file_name) {
	// define variables
	$arr = array();
	$temp_arr = array();

	// read the handle into a buffer for our array
	if($handle = fopen($file_name, "r")) {
		while (!feof($handle)) {
			$buffer = fgets($handle, 8192);
			// delete any comment lines
			if ($buffer[0] == "#") {
				unset($buffer);
			}

			// now, expand our buffer so each word becomes an array element
			if (isset($buffer)) {
				$temp_arr[] = explode(" ", $buffer);
			}
		}
	}

	// explode() creates a new dimension within the array. We need to flatten it
	$temp_arr = array_2d_to_1d($temp_arr);
	return $temp_arr;
}
?>

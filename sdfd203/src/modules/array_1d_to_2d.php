<?php
// modules/array_1d_to_2d.php, Jake Deery, 2018

// flatten an array
function array_2d_to_1d ($input_array) {
	// define variables
	$temp_array = array();
	for ($i = 0; $i < count($input_array); $i++) {
		for ($j = 0; $j < count($input_array[$i]); $j++) {
			$temp_array[] = $input_array[$i][$j];
		}
	}

	return $temp_array;
}
?>

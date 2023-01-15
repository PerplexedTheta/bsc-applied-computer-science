<?php
// modules/update.php, Jake Deery, 2018
if(!function_exists("read_file")) {
	include "read_file.php";
}

if(!function_exists("truncate")) {
	include "truncate.php";
}

if(!function_exists("strip_string")) {
	include "strip_string.php";
}

function update() {
	// define connection details
	$con = mysqli_connect("127.0.0.1", "username", "passwd", "dbname");
	// check connection
	if (mysqli_connect_errno() == true) {
		printf("<SPAN STYLE='color:red;margin:0'>ERROR: Connect failed: " . mysqli_connect_error() . "</SPAN>\n");
		exit();
	}

	// emptry the table of existing rows
	truncate();
	// list our dir into an array, removing the unix periods
	$file_list = scandir("search_source/");
	$clean_list = array_diff($file_list, array('.', '..'));

	// loop through the clean list, each time making the value mach the array
	foreach ($clean_list as $filename_value) {
		// set arr as array (inside the loop so it cleans up each time!)
		$arr = array();
		$temp_arr = read_file("search_source/" . $filename_value);
		// we need to strip all our punctuation from our buffer to make it insertable
		foreach ($temp_arr as $key => $temp_value) {
			$temp_value = strip_string($temp_value);

			// as we're done modifying our array's structure we can store it in the permenant arr
			$arr[] = $temp_value;
		}

		// next, we want to remove useless words like the ones found in illegal.txt
		foreach (read_file("illegal.txt") as $illegal_value) {
			foreach ($arr as $key => $value) {
				if ($value == $illegal_value) {
					unset($arr[$key]);
				}
			}
		}

		// place our array something mysql can comprehend
		$escaped_values = array_map(array($con, "real_escape_string"), $arr); // the escaped values stop any prospect of database injections being done from the outside world
		// place our string into the database
		foreach ($arr as $keyword_values) {
			$insert = "INSERT INTO `axessta_sdfd203`.`searchSchema` (`id`, `keywords`, `directory`) VALUES (NULL, '$keyword_values', '$filename_value');";
			if (!mysqli_query($con, $insert)) {
				printf("<SPAN STYLE='color:red;margin:0'>ERROR: " . $insert . "<BR />"  . mysqli_error($con) . "</SPAN>\n");
				exit();
			} else {
				printf("<SPAN STYLE='color:green;margin:0'>INFO: File $filename_value has keyword $keyword_values. It was added to the database without error.</SPAN>\n");
			}
		}
	}

	mysqli_close($con); // Closing connection
}
?>

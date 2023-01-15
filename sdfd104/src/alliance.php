<!-- alliance.php - Jake Deery -->
<?php
include('modules/session.php');

// get the id
$allianceID = $_GET['id'];

// do the queries
$alliance_query = mysqli_query($con, "SELECT * FROM `alliance` WHERE `allianceID` = '" . $allianceID . "'");
$alliance_row = mysqli_fetch_row($alliance_query);

$headCorp_query = mysqli_query($con, "SELECT * FROM `corporation` WHERE `corporationID` = '" . $alliance_row[1] . "'");
$headCorp_row = mysqli_fetch_row($headCorp_query);
?>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8" />
        <title>SDFD104 BY JAKE DEERY | ALLIANCE SHEET</title>
    </head>
    <body>
        <h1>SDFD104 BY JAKE DEERY | ALLIANCE SHEET</h1>
        <p><b>Alliance name: </b><?php echo $alliance_row[2]; ?></p>
        <p><b>Head corporation: </b><?php printf("<a href='corp.php?id=" . $headCorp_row[0] . "'>" . $headCorp_row[3] . "</a>"); ?></p>
        <p><b>Description: </b><?php echo $alliance_row[3]; ?></p>
        <p><b>Ticker name: </b><?php echo $alliance_row[4]; ?></p>
        <p><b>Member count: </b><?php echo $alliance_row[5]; ?></p>
        <p><b>Openly recruiting: </b><?php
            if($alliance_row[6] == 0) {
            echo ' N'; } else {
            echo ' y';}
            ?></p>
        <hr />
        <p><b>Options: </b><a href="javascript:history.go(-1);">Go back</a> - <a href="profile.php">Return to profile</a> - <a href="modules/logout.php">Logout</a></p>
    </body>
</html>

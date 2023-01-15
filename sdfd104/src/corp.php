<!-- corp.php - Jake Deery -->
<?php
include('modules/session.php');

// get the id
$corporationID = $_GET['id'];

// do the queries
$corporation_query = mysqli_query($con, "SELECT * FROM `corporation` WHERE `corporationID` = '" . $corporationID . "'");
$corporation_row = mysqli_fetch_row($corporation_query);

$ceo_query = mysqli_query($con, "SELECT `charName` FROM `characters` WHERE `characterID` = '" . $corporation_row[1] . "'");
$ceo_row = mysqli_fetch_row($ceo_query);

$alliance_query = mysqli_query($con, "SELECT * FROM `alliance` WHERE `allianceID` = '" . $corporation_row[2] . "'");
$alliance_row = mysqli_fetch_row($alliance_query);
?>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8" />
        <title>SDFD104 BY JAKE DEERY | CORPORATION SHEET</title>
    </head>
    <body>
        <h1>SDFD104 BY JAKE DEERY | CORPORATION SHEET</h1>
        <p><b>Corporation name: </b><?php echo $corporation_row[3]; ?></p>
        <p><b>CEO: </b><?php echo $ceo_row[0]; ?></p>
        <p><b>Member of: </b><?php if($corporation_row[2] == NULL) {
    echo("Not associated with any alliance");
} else {
    printf("<a href='alliance.php?id=" . $alliance_row[0] . "'>" . $alliance_row[2] . "</a>");
}?></p>
        <p><b>Description: </b><?php echo $corporation_row[4]; ?></p>
        <p><b>Ticker name: </b><?php echo $corporation_row[5]; ?></p>
        <p><b>Member count: </b><?php echo $corporation_row[6]; ?></p>
        <p><b>Wallet: </b><?php echo $corporation_row[7]; ?></p>
        <p><b>Openly recruiting: </b><?php
            if($corporation_row[8] == 0) {
            echo ' N'; } else {
            echo ' y';}
            ?></p>
        <hr />
        <p><b>Options: </b><a href="javascript:history.go(-1);">Go back</a> - <a href="profile.php">Return to profile</a> - <a href="modules/logout.php">Logout</a></p>
    </body>
</html>
